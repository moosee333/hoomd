// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifdef NVGRAPH_AVAILABLE

#include "UpdaterClustersGPU.cuh"

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

#include <queue>

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

struct my_int_as_float : public thrust::unary_function<int, float>
    {
    __device__
    float operator()(const int& i) const
        {
        return __int_as_float(i);
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

struct is_reachable : public thrust::unary_function<int, bool>
    {
    __host__ __device__
    bool operator()(const int& i) const
        {
        return i != 2147483647; // 2^31-1
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
    // input matrix in COO format
    nvgraphCOOTopology32I_t COO_input;
    COO_input = (nvgraphCOOTopology32I_t) malloc(sizeof(struct nvgraphCOOTopology32I_st));

    COO_input->nvertices = N;
    COO_input->nedges = 2*n_elements;  // for undirected graph
    COO_input->tag = NVGRAPH_UNSORTED;

    float *d_edge_data_coo;
    float *d_edge_data_csr;

    // allocate COO matrix topology
    check_cuda(cudaMalloc((void **)&(COO_input->source_indices), COO_input->nedges*sizeof(int)));
    check_cuda(cudaMalloc((void **)&(COO_input->destination_indices), COO_input->nedges*sizeof(int)));

    // fill sparse matrix and make it symmetric
    thrust::device_ptr<const uint2> adj(d_adj);
    auto source = thrust::make_transform_iterator(adj, get_source());
    auto destination = thrust::make_transform_iterator(adj, get_destination());
    thrust::device_ptr<int> coo_source(COO_input->source_indices);
    thrust::device_ptr<int> coo_destination(COO_input->destination_indices);

    thrust::copy(source, source+n_elements, coo_source);
    thrust::copy(destination, destination+n_elements, coo_destination);

    // transpose
    thrust::copy(source, source+n_elements, coo_destination+n_elements);
    thrust::copy(destination, destination+n_elements, coo_source+n_elements);

    nvgraphCSRTopology32I_t CSR_output;
    CSR_output = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

    // allocate CSR matrix topology
    check_cuda(cudaMalloc((void **)&(CSR_output->source_offsets), (COO_input->nvertices+1)*sizeof(int)));
    check_cuda(cudaMalloc((void **)&(CSR_output->destination_indices), COO_input->nedges*sizeof(int)));

    // allocate edge data
    check_cuda(cudaMalloc((void **)&d_edge_data_coo, COO_input->nedges*sizeof(float)));
    check_cuda(cudaMalloc((void **)&d_edge_data_csr, COO_input->nedges*sizeof(float)));

    // put ones on the elements of the adjacency matrix in COO format
    thrust::device_ptr<float> edge_data_coo(d_edge_data_coo);
    thrust::fill(thrust::cuda::par(alloc), edge_data_coo, edge_data_coo + COO_input->nedges, 1.0);

    // create nvgraph handle
    nvgraphHandle_t nvgraphH;
    check_nvgraph(nvgraphCreate(&nvgraphH));

    // create parent graph object
    nvgraphGraphDescr_t graph;
    check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &graph));

    // convert COO->CSR
    cudaDataType_t edge_dimT = CUDA_R_32F;
    check_nvgraph(nvgraphConvertTopology(
        nvgraphH,
        NVGRAPH_COO_32,
        COO_input,
        d_edge_data_coo,
        &edge_dimT,
        NVGRAPH_CSR_32,
        CSR_output,
        d_edge_data_csr));

    // these variables will track the dimensions of the current subgraph
    unsigned int nverts = CSR_output->nvertices;

    // set graph connectivity and properties
    unsigned int edge_num_sets = 1;

    check_nvgraph(nvgraphSetGraphStructure(
        nvgraphH,
        graph,
        (void *) CSR_output,
        NVGRAPH_CSR_32));
    check_nvgraph(nvgraphAllocateEdgeData(
        nvgraphH,
        graph,
        edge_num_sets,
        &edge_dimT));
    check_nvgraph(nvgraphSetEdgeData(
        nvgraphH,
        graph,
        d_edge_data_csr,
        0 // edge data set 0
        ));

    /* Vertex data
     * set 0: vector to multiply by
     * set 1: vector to add, and output
     * set 2: origin particle index for connected component
     * set 3: distances from BFS traversal
     */
    int vertex_num_sets = 4;
    cudaDataType_t vertex_dimT[vertex_num_sets];
    for (int i = 0; i < vertex_num_sets; ++i)
        vertex_dimT[i] = CUDA_R_32F;

    check_nvgraph(nvgraphAllocateVertexData(
        nvgraphH,
        graph,
        vertex_num_sets,
        vertex_dimT));

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

    // sorted vertex indices for subgraph
    int *d_vertices;
    check_cuda(cudaMalloc((void **)&d_vertices, nverts*sizeof(int)));
    thrust::device_ptr<int> vertices(d_vertices);

    // LHS of matrix vector multiplication y, and additive input vector
    float *d_y;
    check_cuda(cudaMalloc((void **)&d_y, nverts*sizeof(float)));

    thrust::device_ptr<float> y(d_y);
    thrust::device_ptr<float> diag(d_diag);
    thrust::device_ptr<float> x(d_x);

    // y is vertex data 1
    size_t y_index = 1;

    // attach the ascending particle index as vertex data 2
    float *d_ptl_idx;
    check_cuda(cudaMalloc((void **)&d_ptl_idx, nverts*sizeof(float)));
    thrust::device_ptr<float> ptl_idx(d_ptl_idx);
    auto ptl_idx_as_float = thrust::make_transform_iterator(
        thrust::counting_iterator<int>(0),
        my_int_as_float());
    thrust::copy(
        thrust::cuda::par(alloc),
        ptl_idx_as_float,
        ptl_idx_as_float + N,
        ptl_idx);

    size_t ptls_index = 2; // vertex set 2
    check_nvgraph(nvgraphSetVertexData(
        nvgraphH,
        graph,
        d_ptl_idx,
        ptls_index
        ));

    // difference between x shifted by one and x tself
    float *d_delta_x;
    check_cuda(cudaMalloc((void **) &d_delta_x, sizeof(float)*nverts));
    thrust::device_ptr<float> delta_x(d_delta_x);

    // traversal distances as graph vertex data 3
    int *d_distances;
    check_cuda(cudaMalloc((void **) &d_distances, sizeof(int)*nverts));

    thrust::device_ptr<int> distances(d_distances);
    size_t distances_index = 3;

    // one component
    float *d_component;
    check_cuda(cudaMalloc((void **) &d_component, sizeof(float)*nverts));
    thrust::device_ptr<float> component(d_component);

    // wrap the components output vector
    thrust::device_ptr<unsigned int> components(d_components);

    // a queue for subgraphs (BFS over components)
    std::queue<nvgraphGraphDescr_t> Q;

    // push the parent graph
    Q.push(graph);

    num_components = 0;

    // iteratively partition the graph until all connected components are found
    while (!Q.empty())
        {
        // pop the graph handle from the top of the queue
        auto cur_graph = Q.front();
        Q.pop();

        // get current number of vertices and edges
        struct nvgraphCSRTopology32I_st cur_topology;
        cur_topology.source_offsets = NULL;
        cur_topology.destination_indices = NULL;

        check_nvgraph(nvgraphGetGraphStructure(
            nvgraphH,
            cur_graph,
            &cur_topology,
            NULL));

        nverts = cur_topology.nvertices;

        bool done = false;

        if (nverts == 1)
            {
            // we found a single disconnected vertex, skip spectral analysis
            done = true;
            }
        else
            {
            // use eigenvalue decomposition and estimate partitions from discontinuous steps in the eigenvector associated
            // with the smallest eigenvalue of the Laplacian

            printf("> %d %d\n", nverts, cur_topology.nedges);

            /*
             * compute Laplacian L = diag(A.e) - A
             */
            // LHS y = -diag(A.e)
            float h_one = 1.0;
            float h_minusone = -1.0;
            float zero = 0.0;

            // set vertex data for x = e
            check_nvgraph(nvgraphSetVertexData(
                nvgraphH,
                cur_graph,
                d_ones_float,
                x_index
                ));

            check_nvgraph(nvgraphSrSpmv(
                nvgraphH,
                cur_graph,
                0, // edge set
                &h_minusone,
                x_index, // vertex set for multiplication
                &zero, // multiplying value
                y_index, // vertex set for output
                NVGRAPH_PLUS_TIMES_SR));

            // extract result from matrix vector multiplication into d_diag
            check_nvgraph(nvgraphGetVertexData(
                nvgraphH,
                cur_graph,
                (void *) d_diag,
                y_index));

            // -L = A - diag(A.e)
            // since sparse matrix addition (subtraction) is cumbersome we will carry A and -diag(A.e) separately

            /* find the largest eigenvalue of the negative semidefinite matrix -L (== the singular value of L corresponding
               to a connected component) using the power method

                http://docs.nvidia.com/cuda/cusparse/index.html#csrmv_examples
             */

            float lambda = 0.0;
            float lambda_next = 0.0;

            // initial guess x0 = ones
            check_cuda(cudaMemcpy(d_x, d_ones_float, sizeof(float) * nverts, cudaMemcpyDeviceToDevice));

            for (unsigned int ite = 0; ite < max_ites; ite++)
                {
                /* normalize vector
                 * x= x/|x|
                 */
                float nrm2_x;
                check_cublas(cublasSnrm2_v2(cublasH,
                    nverts,
                    d_x,
                    1, // incx
                    &nrm2_x
                    ));

                float one_over_nrm2_x = 1.0 / nrm2_x;
                check_cublas(cublasSscal_v2(cublasH,
                    nverts,
                    &one_over_nrm2_x,
                    d_x,
                    1 // incx
                    ));

                    {
                    printf("==== ITERATION %d\n", ite);
                    printf("norm %f\n", nrm2_x);
                    }

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
                    cur_graph,
                    d_x,
                    x_index
                    ));

                // update vertex data for y
                check_nvgraph(nvgraphSetVertexData(
                    nvgraphH,
                    cur_graph,
                    d_y,
                    y_index
                    ));

                // y=A*x + y
                check_nvgraph(nvgraphSrSpmv(
                    nvgraphH,
                    cur_graph,
                    0, // edge set
                    &h_one,
                    x_index, // vertex set for multiplication
                    &h_one,
                    y_index, // vertex set for addition
                    NVGRAPH_PLUS_TIMES_SR));

                // extract vertex data from graph
                check_nvgraph(nvgraphGetVertexData(
                    nvgraphH,
                    cur_graph,
                    (void *) d_y,
                    y_index));

                /*
                 * lambda = y**T*x
                 */
                check_cublas(cublasSdot_v2(
                    cublasH,
                    nverts,
                    d_x,
                    1, // incx,
                    d_y,
                    1, // incy
                    &lambda_next
                    ));

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
             * x contains eigenvector corresponding to the largest eigenvalue lambda of -L
             */

            // recover the eigenvalues of the positive semidefinite matrix L by transforming in place x-> -x
            thrust::transform(
                thrust::cuda::par(alloc),
                x,
                x + nverts,
                x,
                thrust::negate<float>());

            // fill vertices with ascending sequence
            auto count = thrust::counting_iterator<int>(0);
            thrust::copy(
                thrust::cuda::par(alloc),
                count,
                count + nverts,
                vertices);

            // sort vertex indices by key (ith component of eigenvector)
            thrust::sort_by_key(
                thrust::cuda::par(alloc),
                x,
                x + nverts,
                vertices);

            // determine a jump in x by taking the adjacent difference
            thrust::transform(
                thrust::cuda::par(alloc),
                x + 1,
                x + nverts,
                x,
                delta_x,
                thrust::minus<float>()
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
                {
                // we found a candidate for a connected component, check if it cannot be split further
                printf("%d %d\n", nverts, cur_topology.nedges);
                nvgraphTraversalParameter_t traversal_param;
                nvgraphTraversalParameterInit(&traversal_param);
                nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index);
                nvgraphTraversalSetUndirectedFlag(&traversal_param, true);

                // do a BFS traversal starting from first index of this putative s.c. component
                int source_vert = 0;
                check_nvgraph(nvgraphTraversal(
                    nvgraphH,
                    cur_graph,
                    NVGRAPH_TRAVERSAL_BFS,
                    &source_vert,
                    traversal_param));

                // extract the distances
                check_nvgraph(nvgraphGetVertexData(
                    nvgraphH,
                    cur_graph,
                    (void *)d_distances,
                    distances_index
                    ));

                /*
                 * if any index is unreachable (distance == 2^31 - 1), this component can still be split
                 */

                // reset vertices to ascending sequence
                thrust::copy(
                    thrust::cuda::par(alloc),
                    count,
                    count + nverts,
                    vertices);

                // sort vertex indices by distance from source vertex
                thrust::sort_by_key(
                    thrust::cuda::par(alloc),
                    distances,
                    distances + nverts,
                    vertices);

                // find first unreachable vertex
                auto unreachable_it = thrust::partition_point(
                    thrust::cuda::par(alloc),
                    distances,
                    distances + nverts,
                    is_reachable()
                    );
                split_idx = unreachable_it - distances;

                if (split_idx == nverts)
                    {
                    // we have found a strongly connected component
                    done = true;
                    }
                } // end if we found a s.c. component candidate

            if (! done)
                {
                // sort the indices to the left of split_idx as required by nvgraph
                thrust::sort(
                    thrust::cuda::par(alloc),
                    vertices,
                    vertices + split_idx
                    );

                // create subgraph object to the left
                nvgraphGraphDescr_t sub_graph_left;
                check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &sub_graph_left));

                // extract the subgraph
                check_nvgraph(nvgraphExtractSubgraphByVertex(
                    nvgraphH,
                    cur_graph,
                    sub_graph_left,
                    d_vertices,
                    split_idx));

                // push the left subgraph in the queue
                Q.push(sub_graph_left);

                // sort the indices to the right of split_idx
                thrust::sort(
                    thrust::cuda::par(alloc),
                    vertices + split_idx,
                    vertices + nverts
                    );

                // create subgraph object to the right
                nvgraphGraphDescr_t sub_graph_right;
                check_nvgraph(nvgraphCreateGraphDescr(nvgraphH, &sub_graph_right));

                // extract the subgraph
                check_nvgraph(nvgraphExtractSubgraphByVertex(
                    nvgraphH,
                    cur_graph,
                    sub_graph_right,
                    d_vertices+split_idx,
                    nverts - split_idx));

                // push the right subgraph in the queue
                Q.push(sub_graph_right);
                }
            } // end if finite connected component

        if (done)
            {
            // extract the particle indices of the connected component
            check_nvgraph(nvgraphGetVertexData(
                nvgraphH,
                cur_graph,
                (void *) d_component,
                ptls_index));

            printf("%p %d\n", d_components, nverts);

            // label the particle indices in the output array by the index of this connected component

            // scatter the component indices as obtained from the vertex data
            auto component_idx = thrust::make_transform_iterator(
                component,
                my_float_as_int());

            auto scatter_it = thrust::make_permutation_iterator(
                components,  // the output vector
                component_idx    // the indices vector
                );

                {
                int *d_indices;
                check_cuda(cudaMalloc(&d_indices, sizeof(int)*nverts));
                thrust::device_ptr<int> indices(d_indices);
                thrust::copy(component_idx, component_idx+ nverts, indices);
                int h_indices[nverts];
                check_cuda(cudaMemcpy(h_indices, d_indices, nverts*sizeof(int),cudaMemcpyDeviceToHost));
                for (unsigned int i = 0; i < nverts; ++i)
                    {
                    int j = h_indices[i];
                    printf("%d\n", j);
                    }
                cudaFree(d_indices);
                }

            thrust::fill(
                thrust::cuda::par(alloc),
                scatter_it,
                scatter_it + nverts,
                num_components++);
            }


        // release this graph descriptor
        check_nvgraph(nvgraphDestroyGraphDescr(nvgraphH, cur_graph));
        };

    // free device data
    check_cuda(cudaFree(d_component));
    check_cuda(cudaFree(d_distances));
    check_cuda(cudaFree(d_ptl_idx));
    check_cuda(cudaFree(d_vertices));
    check_cuda(cudaFree(d_delta_x));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_diag));
    check_cuda(cudaFree(d_ones_float));

    check_cuda(cudaFree(COO_input->source_indices));
    check_cuda(cudaFree(COO_input->destination_indices));
    check_cuda(cudaFree(CSR_output->source_offsets));
    check_cuda(cudaFree(CSR_output->destination_indices));
    check_cuda(cudaFree(d_edge_data_csr));
    check_cuda(cudaFree(d_edge_data_coo));

    // release nvgraph handle
    if (nvgraphH)
        check_nvgraph(nvgraphDestroy(nvgraphH));

    free(COO_input);
    free(CSR_output);

    // clean cublas
    if (cublasH)
        cublasDestroy(cublasH);

    return cudaSuccess;
    }

} // end namespace detail
} // end namespace hpmc
#endif // NVGRAPH_AVAILABLE
