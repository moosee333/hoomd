// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#undef NVGRAPH_AVAILABLE // currently unstable
#ifdef NVGRAPH_AVAILABLE

#include "UpdaterClustersGPU.cuh"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

#include <thrust/random.h>

#include <queue>

#include "hoomd/extern/Eigen/Eigen/Eigenvalues"

#include <cublas_v2.h>
#include <nvgraph.h>

/*! \file UpdaterClustersGPU.cu
    \brief Implements a spectral method for finding strongly connected components
*/

namespace hpmc
{

namespace detail
{

#define check_nvgraph(a) \
    {\
    nvgraphStatus_t status = (a);\
    if ((int)status != NVGRAPH_STATUS_SUCCESS)\
        {\
        printf("nvgraph ERROR %d in file %s line %d\n",status,__FILE__,__LINE__);\
        throw std::runtime_error("Error during clusters update");\
        }\
    }

#define check_cublas(a) \
    {\
    cublasStatus_t status = (a);\
    if ((int)status != CUBLAS_STATUS_SUCCESS)\
        {\
        printf("cublas ERROR %d in file %s line %d\n",status,__FILE__,__LINE__);\
        throw std::runtime_error("Error during clusters update");\
        }\
    }

#define check_cuda(a) \
    {\
    cudaError_t status = (a);\
    if ((int)status != cudaSuccess)\
        {\
        return status;\
        }\
    }

#define check_eigen(a) \
    {\
    Eigen::ComputationInfo info = (a);\
    if (info != Eigen::Success)\
        {\
        printf("Eigen ERROR %d in file %s in line %d\n", info, __FILE__, __LINE__); \
        throw std::runtime_error("Error during clusters update");\
        }\
    }

struct get_source : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__
    unsigned int operator()(const uint2& u) const
        {
        return u.x;
        }
    };

struct get_destination : public thrust::unary_function<uint2, unsigned int>
    {
    __host__ __device__
    unsigned int operator()(const uint2& u) const
        {
        return u.y;
        }
    };

struct my_float_as_int : public thrust::unary_function<float, int>
     {
    __device__
    int operator()(const float& f) const
        {
        return __float_as_int(f);
        }
    };

struct my_int_as_float : public thrust::unary_function<int, float>
     {
    __device__
    float operator()(const int& i) const
        {
        return __int_as_float(i);
        }
    };


struct greater_equal_x : public thrust::unary_function<float, bool>
    {
    __host__ __device__
    greater_equal_x(const float _x)
        : x(_x)
        { }

    __host__ __device__
    bool operator()(const float& f) const
        {
        return f >= x;
        }

    float x;
    };

struct jumps : public thrust::binary_function<float, float, float>
    {
    __device__
    float operator()(const float& a, const float &b) const
        {
        return sqrt((a-b)*(a-b)/fabs(a)/fabs(b));
        }
    };

struct is_reachable : public thrust::unary_function<thrust::tuple<int, int>, bool>
    {
    __host__ __device__
    bool operator()(const thrust::tuple<int,int>& t) const
        {
        return t.get<0>() != 2147483647; // 2^31-1
        }
    };

struct generate_uniform
    {
    generate_uniform(int _seed)
        : seed(_seed)
        { }

    int seed;

    __device__
    float operator () (int idx)
        {
        thrust::minstd_rand randEng(seed);
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
        }
    };

//! Generates edges such that the lower triangular adjacency matrix is filled
struct sort_pair : public thrust::unary_function<uint2, uint2>
    {
    __device__
    uint2 operator()(const uint2& p) const
        {
        unsigned int i = p.x;
        unsigned int j = p.y;
        return (j < i) ? make_uint2(i,j) : make_uint2(j,i);
        }
    };

struct pair_less : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const
        {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
        }
    };

struct pair_equal : public thrust::binary_function<uint2, uint2, bool>
    {
    __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const
        {
        return lhs.x == rhs.x && lhs.y == rhs.y;
        }
    };


cudaError_t gpu_connected_components(
    const uint2 *d_adj,
    unsigned int N,
    unsigned int n_elements,
    unsigned int *d_components,
    unsigned int &num_components,
    cudaStream_t stream,
    unsigned int max_ites,
    float tol,
    float jump_tol,
    const CachedAllocator& alloc)
    {
    // make a copy of the input
    uint2 *d_adj_copy;
    check_cuda(cudaMalloc((void **)&d_adj_copy,n_elements*sizeof(uint2)));
    check_cuda(cudaMemcpy(d_adj_copy, d_adj, sizeof(uint2)*n_elements,cudaMemcpyDeviceToDevice));

    // fill sparse matrix and make it symmetric
    thrust::device_ptr<uint2> adj_copy(d_adj_copy);

    // sort every pair
    thrust::transform(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        adj_copy,
        sort_pair());

    // sort the list of pairs
    thrust::sort(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        pair_less());

    // remove duplicates
    auto unique_end = thrust::unique(
        thrust::cuda::par(alloc),
        adj_copy,
        adj_copy + n_elements,
        pair_equal());

    unsigned int n_unique = unique_end - adj_copy;

    unsigned int *d_label;
    check_cuda(cudaMalloc((void **)&d_label,N*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> label(d_label);

    // label all endpoints of edges
    auto source = thrust::make_transform_iterator(adj_copy, get_source());
    auto destination = thrust::make_transform_iterator(adj_copy, get_destination());

    auto one_it = thrust::constant_iterator<unsigned int>(1);
    thrust::fill(
        thrust::cuda::par(alloc),
        label,
        label + N,
        0);

    thrust::scatter(
        thrust::cuda::par(alloc),
        one_it,
        one_it + n_unique,
        source,
        label);

    thrust::scatter(
        thrust::cuda::par(alloc),
        one_it,
        one_it + n_unique,
        destination,
        label);

    // partition the vertices into trivial ones, i.e. those not connected to any edge, and connected ones
    int *d_connected_vertices;
    int *d_trivial_components;
    check_cuda(cudaMalloc((void **)&d_connected_vertices,N*sizeof(int)));
    thrust::device_ptr<int> connected_vertices(d_connected_vertices);

    check_cuda(cudaMalloc((void **)&d_trivial_components,N*sizeof(int)));
    thrust::device_ptr<int> trivial_components(d_trivial_components);

    auto part = thrust::stable_partition_copy(
        thrust::cuda::par(alloc),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(0)+N,
        label,
        connected_vertices,
        trivial_components,
        thrust::identity<int>()
        );
    unsigned int n_connected = part.first - connected_vertices;
    unsigned int n_trivial = part.second - trivial_components;

    // input matrix in COO format
    nvgraphCOOTopology32I_t COO_input;
    COO_input = (nvgraphCOOTopology32I_t) malloc(sizeof(struct nvgraphCOOTopology32I_st));

    COO_input->nvertices = N;
    COO_input->nedges = 2*n_unique;  // for undirected graph
    COO_input->tag = NVGRAPH_UNSORTED;

    // allocate COO matrix topology
    check_cuda(cudaMalloc((void **)&(COO_input->source_indices), COO_input->nedges*sizeof(int)));
    check_cuda(cudaMalloc((void **)&(COO_input->destination_indices), COO_input->nedges*sizeof(int)));

    thrust::device_ptr<int> coo_source(COO_input->source_indices);
    thrust::device_ptr<int> coo_destination(COO_input->destination_indices);

    thrust::copy(source, source+n_unique, coo_source);
    thrust::copy(destination, destination+n_unique, coo_destination);

    // transpose
    thrust::copy(source, source+n_unique, coo_destination+n_unique);
    thrust::copy(destination, destination+n_unique, coo_source+n_unique);

    // a current limitation of nvgraph is to only support graphs with a single edge/vertex data type
    // we need both float (for spectral analysis) and int (for traversal), so create two parallel graphs
    nvgraphCSRTopology32I_t CSR_output_F;
    nvgraphCSRTopology32I_t CSR_output_I;
    CSR_output_F = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));
    CSR_output_I = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

    // allocate CSR matrix topology
    check_cuda(cudaMalloc((void **)&(CSR_output_F->source_offsets), (COO_input->nvertices+1)*sizeof(int)));
    check_cuda(cudaMalloc((void **)&(CSR_output_F->destination_indices), COO_input->nedges*sizeof(int)));

    check_cuda(cudaMalloc((void **)&(CSR_output_I->source_offsets), (COO_input->nvertices+1)*sizeof(int)));
    check_cuda(cudaMalloc((void **)&(CSR_output_I->destination_indices), COO_input->nedges*sizeof(int)));

    // allocate edge data
    float *d_edge_data_coo_F;
    float *d_edge_data_csr_F;

    check_cuda(cudaMalloc((void **)&d_edge_data_coo_F, COO_input->nedges*sizeof(float)));
    check_cuda(cudaMalloc((void **)&d_edge_data_csr_F, COO_input->nedges*sizeof(float)));

    // put ones on the elements of the adjacency matrix in COO format
    thrust::device_ptr<float> edge_data_coo_F(d_edge_data_coo_F);

    thrust::fill(thrust::cuda::par(alloc), edge_data_coo_F, edge_data_coo_F + COO_input->nedges, 1.0);

    // create nvgraph handle
    nvgraphHandle_t nvgraphH;
    check_nvgraph(nvgraphCreate(&nvgraphH));

    // create parent graph objects
    nvgraphGraphDescr_t graph_F;
    nvgraphGraphDescr_t graph_I;
    check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &graph_F));
    check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &graph_I));

    // convert COO->CSR
    cudaDataType_t edge_dimT_F = CUDA_R_32F;
    cudaDataType_t edge_dimT_I = CUDA_R_32I;
    check_nvgraph(nvgraphConvertTopology(
        nvgraphH,
        NVGRAPH_COO_32,
        COO_input,
        d_edge_data_coo_F,
        &edge_dimT_F,
        NVGRAPH_CSR_32,
        CSR_output_F,
        d_edge_data_csr_F));

    /*
    check_nvgraph(nvgraphConvertTopology(
        nvgraphH,
        NVGRAPH_COO_32,
        COO_input,
        d_edge_data_coo_I,
        &edge_dimT_I,
        NVGRAPH_CSR_32,
        CSR_output_I,
        d_edge_data_csr_I));
    */
    // copy over topology
    CSR_output_I->nvertices = CSR_output_F->nvertices;
    CSR_output_I->nedges = CSR_output_F->nedges;
    check_cuda(cudaMemcpy(CSR_output_I->source_offsets,CSR_output_F->source_offsets,(CSR_output_F->nvertices+1)*sizeof(int),cudaMemcpyDeviceToDevice));
    check_cuda(cudaMemcpy(CSR_output_I->destination_indices,CSR_output_F->destination_indices,(CSR_output_F->nedges)*sizeof(int),cudaMemcpyDeviceToDevice));

    // these variables will track the dimensions of the current subgraph
    unsigned int nverts = CSR_output_F->nvertices;
    unsigned int nedges = CSR_output_F->nedges;

    // set graph connectivity and properties
    unsigned int edge_num_sets_F = 1;
    unsigned int edge_num_sets_I = 0;

    check_nvgraph(nvgraphSetGraphStructure(
        nvgraphH,
        graph_F,
        (void *) CSR_output_F,
        NVGRAPH_CSR_32));
    check_nvgraph(nvgraphAllocateEdgeData(
        nvgraphH,
        graph_F,
        edge_num_sets_F,
        &edge_dimT_F));
    check_nvgraph(nvgraphSetEdgeData(
        nvgraphH,
        graph_F,
        d_edge_data_csr_F,
        0 // edge data set 0
        ));

    check_nvgraph(nvgraphSetGraphStructure(
        nvgraphH,
        graph_I,
        (void *) CSR_output_I,
        NVGRAPH_CSR_32));
    #if 0
    check_nvgraph(nvgraphAllocateEdgeData(
        nvgraphH,
        graph_I,
        edge_num_sets_I,
        &edge_dimT_I));
    #endif


    /* Vertex data
     * F:
     * set 0: vector to multiply by (float)
     * set 1: vector to add, and output (float)
     * set 2: origin particle index for connected component (int as float)
     *
     * I:
     * set 0: distances from BFS traversal (int)
     */
    int vertex_num_sets_F = 3;
    cudaDataType_t vertex_dimT_F[vertex_num_sets_F];
    vertex_dimT_F[0] = CUDA_R_32F;
    vertex_dimT_F[1] = CUDA_R_32F;
    vertex_dimT_F[2] = CUDA_R_32F;

    int vertex_num_sets_I = 1;
    cudaDataType_t vertex_dimT_I[vertex_num_sets_I];
    vertex_dimT_I[0] = CUDA_R_32I;

    check_nvgraph(nvgraphAllocateVertexData(
        nvgraphH,
        graph_F,
        vertex_num_sets_F,
        vertex_dimT_F));

    check_nvgraph(nvgraphAllocateVertexData(
        nvgraphH,
        graph_I,
        vertex_num_sets_I,
        vertex_dimT_I));

    // set up cublas handle
    cublasHandle_t cublasH = NULL;

    check_cublas(cublasCreate(&cublasH));
    check_cublas(cublasSetStream(cublasH, stream));

    // a RHS vector of ones
    float *d_ones_float;
    check_cuda(cudaMalloc((void **)&d_ones_float, nverts*sizeof(float)));
    thrust::device_ptr<float> ones_float(d_ones_float);
    thrust::fill(ones_float, ones_float + nverts, 1.0);

    // stores the diagonal matrix
    float *d_diag;
    check_cuda(cudaMalloc((void **)&d_diag, nverts*sizeof(float)));

    // solution vector
    float *d_x;
    check_cuda(cudaMalloc((void **) &d_x, nverts*sizeof(float)));
    size_t x_index = 0; // vertex data 0

    // vector from last iteration
    float *d_last_x;
    check_cuda(cudaMalloc((void **) &d_last_x, nverts*sizeof(float)));

    // sorted vertex indices for subgraph
    int *d_vertices_left;
    check_cuda(cudaMalloc((void **)&d_vertices_left, nverts*sizeof(int)));
    thrust::device_ptr<int> vertices_left(d_vertices_left);

    int *d_vertices_right;
    check_cuda(cudaMalloc((void **)&d_vertices_right, nverts*sizeof(int)));
    thrust::device_ptr<int> vertices_right(d_vertices_right);

    // LHS of matrix vector multiplication y, and additive input vector
    float *d_y;
    check_cuda(cudaMalloc((void **)&d_y, nverts*sizeof(float)));

    // the eigenvector with smallest eigenvalue
    float *d_z;
    check_cuda(cudaMalloc((void **)&d_z, nverts*sizeof(float)));

    thrust::device_ptr<float> y(d_y);
    thrust::device_ptr<float> z(d_z);
    thrust::device_ptr<float> diag(d_diag);
    thrust::device_ptr<float> x(d_x);

    // y is vertex data 1
    size_t y_index = 1;

    // dense matrix of Lanzcos vectors, in column major
    float *d_V;
    check_cuda(cudaMalloc((void **)&d_V, nverts*max_ites*sizeof(float)));

    // RHS for eigenvector computation
    float *d_v;
    check_cuda(cudaMalloc((void **)&d_v, max_ites*sizeof(float)));

    // attach the ascending particle index as vertex data 2 (float)
    float *d_ptl_idx;
    check_cuda(cudaMalloc((void **)&d_ptl_idx, nverts*sizeof(float)));
    thrust::device_ptr<float> ptl_idx(d_ptl_idx);
    thrust::transform(
        thrust::cuda::par(alloc),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(0) + N,
        ptl_idx,
        my_int_as_float());

    size_t ptls_index = 2; // vertex set 2 on F
    check_nvgraph(nvgraphSetVertexData(
        nvgraphH,
        graph_F,
        (void *)d_ptl_idx,
        ptls_index
        ));

    int *d_source_idx;
    check_cuda(cudaMalloc((void **)&d_source_idx, sizeof(int)));
    thrust::device_ptr<int> source_idx(d_source_idx);

    // difference between x shifted by one and x itself
    float *d_delta_x;
    check_cuda(cudaMalloc((void **) &d_delta_x, sizeof(float)*nverts));
    thrust::device_ptr<float> delta_x(d_delta_x);

    // traversal distances as graph vertex data 0
    int *d_distances;
    check_cuda(cudaMalloc((void **) &d_distances, sizeof(int)*nverts));
    thrust::device_ptr<int> distances(d_distances);
    size_t distances_index = 0;

    // for a component only
    int *d_component_distances;
    check_cuda(cudaMalloc((void **) &d_component_distances, sizeof(int)*nverts));
    thrust::device_ptr<int> component_distances(d_component_distances);

    // one component
    float *d_component;
    check_cuda(cudaMalloc((void **) &d_component, sizeof(float)*nverts));
    thrust::device_ptr<float> component(d_component);

    // wrap the components output vector
    thrust::device_ptr<unsigned int> components(d_components);

    // a queue for subgraphs (BFS over components)
    std::queue<nvgraphGraphDescr_t>  Q;

    // take the subgraph of connected vertices

    // create subgraph objects to the left
    nvgraphGraphDescr_t sub_graph_connected;
    check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &sub_graph_connected));

    // extract the float subgraph
    check_nvgraph(nvgraphExtractSubgraphByVertex(
        nvgraphH,
        graph_F,
        sub_graph_connected,
        d_connected_vertices,
        n_connected));

    // destroy the parent graph already
    check_nvgraph(nvgraphDestroyGraphDescr(nvgraphH, graph_F));

    // push the root graph
    Q.push(sub_graph_connected);

    // already label the trivial components
    thrust::scatter(
        thrust::cuda::par(alloc),
        thrust::counting_iterator<unsigned int>(0),
        thrust::counting_iterator<unsigned int>(0)+n_trivial,
        trivial_components,
        components);

    num_components = n_trivial;

    int seed = 0;

    // iteratively partition the graph until all connected components are found
    while (!Q.empty())
        {
        // pop the graph handle from the top of the queue
        auto cur_graph_F = Q.front();
        Q.pop();

        // get current number of vertices and edges
        struct nvgraphCSRTopology32I_st cur_topology;
        cur_topology.source_offsets = NULL;
        cur_topology.destination_indices = NULL;

        check_nvgraph(nvgraphGetGraphStructure(
            nvgraphH,
            cur_graph_F,
            &cur_topology,
            NULL));

        nverts = cur_topology.nvertices;
        nedges = cur_topology.nedges;

        bool done = false;

        if (nedges == 0)
            {
            // we found N disconnected vertices, finalize
            done = true;
            }

        #if 0
        else
            {
            // use eigenvalue decomposition and estimate partitions from discontinuous steps in the eigenvector associated
            // with the smallest eigenvalue of the Laplacian

            /*
             * compute Laplacian L = diag(A.e) - A
             */
            // LHS y = diag(A.e)
            float h_one = 1.0;
            float h_minusone = -1.0;
            float zero = 0.0;

            // set vertex data for x = e
            check_nvgraph(nvgraphSetVertexData(
                nvgraphH,
                cur_graph_F,
                d_ones_float,
                x_index
                ));

            check_nvgraph(nvgraphSrSpmv(
                nvgraphH,
                cur_graph_F,
                0, // edge set
                &h_one,
                x_index, // vertex set for multiplication
                &zero, // multiplying value
                y_index, // vertex set for output
                NVGRAPH_PLUS_TIMES_SR));

            // extract result from matrix vector multiplication into d_diag
            check_nvgraph(nvgraphGetVertexData(
                nvgraphH,
                cur_graph_F,
                (void *) d_diag,
                y_index));

            // Laplacian L = diag(A.e) - A
            // since sparse matrix addition (subtraction) is cumbersome we will carry -A and diag(A.e) separately

            /* find the largest eigenvalue of the negative semidefinite matrix -L (== the singular value of L corresponding
               to a connected component) using the Lanczos algorithm
             */

            float lambda = 0.0;
            float lambda_next = 0.0;

            float alpha[max_ites];
            float beta[max_ites];

            /*
             * set up the first iteration
             */

            // initial vector x0 = ones
            check_cuda(cudaMemcpy(d_x, d_ones_float, sizeof(float) * nverts, cudaMemcpyDeviceToDevice));

            /* normalize vector
             * x= x/|x|
             */
            float one_over_nrm2_x = 1.0 / sqrtf(nverts);
            check_cublas(cublasSscal_v2(cublasH,
                nverts,
                &one_over_nrm2_x,
                d_x,
                1 // incx
                ));

            // save first Lanczos vector
            cudaMemcpy(&d_V[0], d_x, sizeof(float)*nverts, cudaMemcpyDeviceToDevice);

            /*
             * y = L*x = diag(A.e)*x - A*x
             */

            // y = diag(A.e)*x (component wise multiplication)
            thrust::transform(
                x,
                x + nverts,
                diag,
                y,
                thrust::multiplies<float>()
                );

            // update vertex data for x
            check_nvgraph(nvgraphSetVertexData(
                nvgraphH,
                cur_graph_F,
                d_x,
                x_index
                ));

            // update vertex data for y
            check_nvgraph(nvgraphSetVertexData(
                nvgraphH,
                cur_graph_F,
                d_y,
                y_index
                ));

            // y'=A*x + y
            check_nvgraph(nvgraphSrSpmv(
                nvgraphH,
                cur_graph_F,
                0, // edge set
                &h_minusone,
                x_index, // vertex set for multiplication
                &h_one,
                y_index, // vertex set for addition
                NVGRAPH_PLUS_TIMES_SR));

            // extract vertex data from graph
            check_nvgraph(nvgraphGetVertexData(
                nvgraphH,
                cur_graph_F,
                (void *) d_y,
                y_index));

            // alpha = y'.x
            alpha[0] = 0.0;
            check_cublas(cublasSdot_v2(
                cublasH,
                nverts,
                d_x,
                1, // incx,
                d_y,
                1, // incy
                &alpha[0]
                ));

            // y = y' - alpha*x
            float minus_alpha = -alpha[0];
            check_cublas(cublasSaxpy_v2(
                cublasH,
                nverts,
                &minus_alpha,
                d_x,
                1, // incx
                d_y,
                1  // incy
                ));

            // store result in x
            check_cuda(cudaMemcpy(d_x, d_y, nverts*sizeof(float), cudaMemcpyDeviceToDevice));

            for (unsigned int ite = 1; ite < max_ites; ite++)
                {
                /*
                 * beta = |x|
                 */
                beta[ite] = 0.0;
                check_cublas(cublasSnrm2_v2(cublasH,
                    nverts,
                    d_x,
                    1, // incx
                    &beta[ite]
                    ));

                if (beta[ite] != 0.0)
                    {
                    // x_i = (x_i-1)/beta
                    float one_over_beta = 1.0 / beta[ite];
                    check_cublas(cublasSscal_v2(cublasH,
                        nverts,
                        &one_over_beta,
                        d_x,
                        1 // incx
                        ));
                    }
                else
                    {
                    // draw a random starting vector to decrease the chance it is orthogonal to an eigenvector
                    auto count = thrust::make_counting_iterator<int>(0);
                    thrust::transform(
                        thrust::cuda::par(alloc),
                        count,
                        count + nverts,
                        x,
                        generate_uniform(seed++));

                    /* normalize vector
                     * x= x/|x|
                     */
                    float nrm2_x = 0.0;
                    check_cublas(cublasSnrm2_v2(
                        cublasH,
                        nverts,
                        d_x,
                        1, // incx
                        &nrm2_x
                        ));

                    float one_over_norm = 1.0 / nrm2_x;
                    check_cublas(cublasSscal_v2(cublasH,
                        nverts,
                        &one_over_norm,
                        d_x,
                        1 // incx
                        ));
                    }

                // save the Lanczos vector
                check_cuda(cudaMemcpy(&d_V[ite*nverts], d_x, sizeof(float)*nverts, cudaMemcpyDeviceToDevice));

                // y' =  w'
                /*
                 * y = -L*x = A*x - diag(A.e)*x
                 */

                // y = diag(A.e)*x (component wise multiplication)
                thrust::transform(
                    x,
                    x + nverts,
                    diag,
                    y,
                    thrust::multiplies<float>()
                    );

                // update vertex data for x
                check_nvgraph(nvgraphSetVertexData(
                    nvgraphH,
                    cur_graph_F,
                    d_x,
                    x_index
                    ));

                // update vertex data for y
                check_nvgraph(nvgraphSetVertexData(
                    nvgraphH,
                    cur_graph_F,
                    d_y,
                    y_index
                    ));

                // y'=A*x + y
                check_nvgraph(nvgraphSrSpmv(
                    nvgraphH,
                    cur_graph_F,
                    0, // edge set
                    &h_minusone,
                    x_index, // vertex set for multiplication
                    &h_one,
                    y_index, // vertex set for addition
                    NVGRAPH_PLUS_TIMES_SR));

                // extract vertex data from graph
                check_nvgraph(nvgraphGetVertexData(
                    nvgraphH,
                    cur_graph_F,
                    (void *) d_y,
                    y_index));

                // alpha = y'.x
                alpha[ite] = 0.0;
                check_cublas(cublasSdot_v2(
                    cublasH,
                    nverts,
                    d_x,
                    1, // incx,
                    d_y,
                    1, // incy
                    &alpha[ite]
                    ));

                // y = y' - alpha*x
                minus_alpha = -alpha[ite];
                check_cublas(cublasSaxpy_v2(
                    cublasH,
                    nverts,
                    &minus_alpha,
                    d_x,
                    1, // incx
                    d_y,
                    1 // incy
                    ));

                // y = y - beta*x_(i-1)
                float minus_beta = -beta[ite];
                check_cublas(cublasSaxpy_v2(
                    cublasH,
                    nverts,
                    &minus_beta,
                    &d_V[(ite-1)*nverts],
                    1, // incx
                    d_y,
                    1  // incy
                    ));

                /*
                 *solve the eigenvalue problem for the tridiagonal matrix T on the CPU
                 */
                    {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es;
                    Eigen::Map<Eigen::VectorXf> diag(alpha,ite+1);
                    Eigen::Map<Eigen::VectorXf> subdiag(beta+1,ite);
                    es.computeFromTridiagonal(diag, subdiag);
                    check_eigen(es.info());

                    // get the eigenvalues
                    auto eval = es.eigenvalues();

                    // eigenvalue of largest magnitude
                    lambda_next = eval(ite);

                    // get the eigenvectors, sorted by increasing eigenvalue
                    auto evec = es.eigenvectors();

                    // copy eigenvector corresponding to smallest eigenvalue to device
                    float max_evec[ite];
                    for (unsigned int j = 0; j < ite + 1; ++j)
                        max_evec[j] = evec(0,j);
                    check_cuda(cudaMemcpy(d_v, &max_evec[0], sizeof(float)*(ite+1), cudaMemcpyHostToDevice));

                    // back out the corresponding eigenvector of L
                    check_cublas(cublasSgemv_v2(
                        cublasH,
                        CUBLAS_OP_N,
                        nverts,
                        ite+1,
                        &h_one,
                        d_V,
                        nverts,
                        d_v,
                        1, // incx
                        &zero,
                        d_z,
                        1 // incy
                        ));
                    }

                /*
                 * check if convergence
                 */
                if ( (ite > 0) && fabs(lambda - lambda_next) < tol)
                    break;

                /*
                 * x := y
                 * lambda = lambda_next
                 */
                check_cuda(cudaMemcpy(d_x, d_y, nverts*sizeof(float), cudaMemcpyDeviceToDevice));
                lambda = lambda_next;
                } // end of an iteration of the power method

            /*
             * z contains eigenvector corresponding to the largest eigenvalue lambda of -L
             */

            // fill vertices with ascending sequence
            auto count = thrust::make_counting_iterator<int>(0);
            thrust::copy(
                thrust::cuda::par(alloc),
                count,
                count + nverts,
                vertices);

            // sort vertex indices by key (ith component of eigenvector)
            thrust::sort_by_key(
                thrust::cuda::par(alloc),
                z,
                z + nverts,
                vertices);

            // determine a jump in x by taking the adjacent difference
            thrust::transform(
                thrust::cuda::par(alloc),
                z + 1,
                z + nverts,
                z,
                delta_x,
                jumps()
                );

            // pick up the jump at i
            auto jump_it = thrust::find_if(
                thrust::cuda::par(alloc),
                delta_x,
                delta_x + nverts - 1,
                greater_equal_x(jump_tol)
                );

            // last index of subgraph + 1
            unsigned int split_idx = (jump_it - delta_x) + 1;

            if (split_idx == nverts)
        #else
        unsigned int n_left = 0;
        unsigned int n_right = 0;
        if (!done)
        #endif
                {
                // we found a candidate for a connected component, check if it cannot be split further
                nvgraphTraversalParameter_t traversal_param;
                nvgraphTraversalParameterInit(&traversal_param);
                nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
//                nvgraphTraversalSetUndirectedFlag(&traversal_param, 1);

                // do a BFS traversal starting from first index of this putative s.c. component

                // download the component
                check_nvgraph(nvgraphGetVertexData(
                    nvgraphH,
                    cur_graph_F,
                    (void *) d_component,
                    ptls_index));

                auto component_idx = thrust::make_transform_iterator(
                    component,
                    my_float_as_int());

                // transfer to host
                int source_vert;
                thrust::copy(
                    thrust::cuda::par(alloc),
                    component_idx,
                    component_idx + 1,
                    source_idx);

                check_cuda(cudaMemcpy(&source_vert, d_source_idx, sizeof(int), cudaMemcpyDeviceToHost));

                // do the traversal
                check_nvgraph(nvgraphTraversal(
                    nvgraphH,
                    graph_I,
                    NVGRAPH_TRAVERSAL_BFS,
                    &source_vert,
                    traversal_param));

                // extract the distances
                check_nvgraph(nvgraphGetVertexData(
                    nvgraphH,
                    graph_I,
                    (void *)d_distances,
                    distances_index
                    ));

                /*
                 * if any index in the component is unreachable (distance == 2^31 - 1), it can still be split
                 */
                auto distance_transform = thrust::make_permutation_iterator(
                        distances,
                        component_idx);

                thrust::copy(
                    thrust::cuda::par(alloc),
                    distance_transform,
                    distance_transform + nverts,
                    component_distances);

                // find first unreachable vertex
                auto zipit = thrust::make_zip_iterator(thrust::make_tuple(
                    component_distances, thrust::counting_iterator<int>(0)));

                auto zip_left = thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::discard_iterator<int>(),
                    vertices_left));
                auto zip_right = thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::discard_iterator<int>(),
                    vertices_right));

                auto part_reachable = thrust::stable_partition_copy(
                    thrust::cuda::par(alloc),
                    zipit,
                    zipit + nverts,
                    zip_left,
                    zip_right,
                    is_reachable()
                    );
                n_left = part_reachable.first - zip_left;
                n_right = part_reachable.second - zip_right;

                if (n_left == nverts)
                    {
                    // we have found a strongly connected component
                    done = true;
                    }
                } // end if we found a s.c. component candidate


            if (! done)
                {
                // create subgraph objects to the left
                nvgraphGraphDescr_t sub_graph_left_F;
                check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &sub_graph_left_F));

                // extract the float subgraph
                check_nvgraph(nvgraphExtractSubgraphByVertex(
                    nvgraphH,
                    cur_graph_F,
                    sub_graph_left_F,
                    d_vertices_left,
                    n_left));

                // push the left subgraph in the queue
                Q.push(sub_graph_left_F);

                // create subgraph object to the right
                nvgraphGraphDescr_t sub_graph_right_F;
                check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &sub_graph_right_F));

                // extract the float subgraph
                check_nvgraph(nvgraphExtractSubgraphByVertex(
                    nvgraphH,
                    cur_graph_F,
                    sub_graph_right_F,
                    d_vertices_right,
                    n_right));

                // push the right subgraph in the queue
                Q.push(sub_graph_right_F);
                }
        #if 0
            } // end if finite connected component
        #endif
        if (done)
            {
            // extract the particle indices of the connected component
            check_nvgraph(nvgraphGetVertexData(
                nvgraphH,
                cur_graph_F,
                (void *) d_component,
                ptls_index));

            // label the particle indices in the output array by the index of this connected component

            // scatter the component indices as obtained from the vertex data
            auto component_idx = thrust::make_transform_iterator(
                component,
                my_float_as_int());

            auto scatter_it = thrust::make_permutation_iterator(
                components,  // the output vector
                component_idx    // the indices vector
                );

            thrust::fill(
                thrust::cuda::par(alloc),
                scatter_it,
                scatter_it + nverts,
                num_components++);
            }


        // release this graph descriptor
        check_nvgraph(nvgraphDestroyGraphDescr(nvgraphH, cur_graph_F));
        };

    // free device data
    check_cuda(cudaFree(d_component));
    check_cuda(cudaFree(d_distances));
    check_cuda(cudaFree(d_component_distances));
    check_cuda(cudaFree(d_ptl_idx));
    check_cuda(cudaFree(d_source_idx));
    check_cuda(cudaFree(d_vertices_left));
    check_cuda(cudaFree(d_vertices_right));
    check_cuda(cudaFree(d_delta_x));
    check_cuda(cudaFree(d_z));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_last_x));
    check_cuda(cudaFree(d_diag));
    check_cuda(cudaFree(d_ones_float));
    check_cuda(cudaFree(d_V));
    check_cuda(cudaFree(d_v));

    check_cuda(cudaFree(COO_input->source_indices));
    check_cuda(cudaFree(COO_input->destination_indices));
    check_cuda(cudaFree(CSR_output_F->source_offsets));
    check_cuda(cudaFree(CSR_output_F->destination_indices));
    check_cuda(cudaFree(CSR_output_I->source_offsets));
    check_cuda(cudaFree(CSR_output_I->destination_indices));
    check_cuda(cudaFree(d_edge_data_csr_F));
    check_cuda(cudaFree(d_edge_data_coo_F));
    check_cuda(cudaFree(d_adj_copy));
    check_cuda(cudaFree(d_label));
    check_cuda(cudaFree(d_connected_vertices));
    check_cuda(cudaFree(d_trivial_components));

    check_nvgraph(nvgraphDestroyGraphDescr(nvgraphH, graph_I));

    // release nvgraph handle
    if (nvgraphH)
        check_nvgraph(nvgraphDestroy(nvgraphH));

    free(COO_input);
    free(CSR_output_F);
    free(CSR_output_I);

    // clean cublas
    if (cublasH)
        cublasDestroy(cublasH);

    return cudaSuccess;
    }

} // end namespace detail
} // end namespace hpmc

#endif // NVGRAPH_AVAILABLE
