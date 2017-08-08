
#ifdef ENABLE_CUDA

template<typename Potential>
COMPForceComputeGPU<Potential>::COMPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<NeighborList> nlist,
                                                    const std::string& log_suffix):
    COMPForceCompute<Potential>(sysdef, nlist, log_suffix), m_param(0),
    m_gpuVertices(this->m_exec_conf), m_typeStarts(this->m_exec_conf), m_typeCounts(this->m_exec_conf)
{
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
    {
        this->m_exec_conf->msg->error() << "Creating a COMPForceComputeGPU with no GPU in the execution configuration"
                                        << std::endl;
        throw std::runtime_error("Error initializing COMPForceComputeGPU");
    }

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
    {
        int s=1;
        while (s <= this->m_exec_conf->dev_prop.warpSize)
        {
            valid_params.push_back(block_size*10000 + s);
            s = s * 2;
        }
    }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "comp_" + Potential::getName(), this->m_exec_conf));
#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(this->m_pdata->getDomainDecomposition());
#endif

    m_precompute = false;
    m_has_been_precomputed = false;
}

template<typename Potential>
void COMPForceComputeGPU<Potential>::rebuildGeometry()
{
    m_typeStarts.resize(this->m_pdata->getNTypes());
    m_typeCounts.resize(this->m_pdata->getNTypes());

    unsigned int vertCount(0);
    for(unsigned int type(0); type < this->m_typeVertices.size(); ++type)
        vertCount += this->m_typeVertices[type].size();

    m_gpuVertices.resize(vertCount);

    vertCount = 0;

    for(unsigned int type(0); type < this->m_typeVertices.size(); ++type)
    {
        m_typeStarts[type] = vertCount;
        m_typeCounts[type] = this->m_typeVertices[type].size();
        for(typename COMPForceCompute<Potential>::Shape::const_iterator vertIter(this->m_typeVertices[type].begin());
            vertIter != this->m_typeVertices[type].end(); ++vertIter, ++vertCount)
        {
            m_gpuVertices[vertCount] = vec_to_scalar3(*vertIter);
        }
    }
}

template<typename Potential>
void COMPForceComputeGPU<Potential>::setVertices(unsigned int typ,
                                                 const pybind11::list &verts)
{
    COMPForceCompute<Potential>::setVertices(typ, verts);

    rebuildGeometry();
}

template<typename Potential>
void COMPForceComputeGPU<Potential>::computeForces(unsigned int timestep)
{
    // start by updating the neighborlist
    if (!m_precompute)
        this->m_nlist->compute(timestep);

    // if we have already computed and the neighbor list remains current do not recompute
    if (!m_precompute && m_has_been_precomputed &&
        !this->m_nlist->hasBeenUpdated(timestep))
        return;

    m_has_been_precomputed = false;

    // start the profile
    if (this->m_prof)
        this->m_prof->push(this->exec_conf, this->m_prof_name);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
    {
        this->m_exec_conf->msg->error() << "COMPForceComputeGPU cannot handle a half neighborlist"
                                        << std::endl;
        throw std::runtime_error("Error computing forces in COMPForceComputeGPU");
    }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_quat(this->m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename Potential::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar3> d_vertices(this->m_gpuVertices, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_vertexTypeStarts(this->m_typeStarts, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_vertexTypeCounts(this->m_typeCounts, access_location::device, access_mode::read);

    GeometryArgs geom_args(this->m_gpuVertices.size(), d_vertices.data, d_vertexTypeStarts.data, d_vertexTypeCounts.data);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    if (! m_param) this->m_tuner->begin();
    unsigned int param = !m_param ?  this->m_tuner->getParam() : m_param;
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    compute_comp_force_gpu<Potential>(
        pair_args_t(d_force.data,
                    d_virial.data,
                    this->m_virial.getPitch(),
                    this->m_pdata->getN(),
                    this->m_pdata->getMaxN(),
                    d_pos.data,
                    d_diameter.data,
                    d_charge.data,
                    box,
                    d_n_neigh.data,
                    d_nlist.data,
                    d_head_list.data,
                    d_rcutsq.data,
                    d_ronsq.data,
                    this->m_nlist->getNListArray().getPitch(),
                    this->m_pdata->getNTypes(),
                    block_size,
                    this->m_shift_mode,
                    flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial],
                    threads_per_particle,
                    this->m_exec_conf->getComputeCapability()/10,
                    this->m_exec_conf->dev_prop.maxTexture1DLinear),
        d_quat.data,
        d_torque.data,
        geom_args,
        d_params.data);

    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    if (!m_param) this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop(this->exec_conf);
}

// Modified to work with pybind11
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
  \tparam T Class type to export. \b Must be an instantiated COMPForceComputeGPU class template.
  \tparam Base Base class of \a T. \b Must be COMPForceCompute<Potential> with the same Potential as used in \a T.
*/
/*
template <class T, class Base> void export_COMPForceComputeGPU(const std::string& name)
{
    boost::python::class_<T, std::shared_ptr<T>, boost::python::bases<Base>, boost::noncopyable >
        (name.c_str(), boost::python::init< std::shared_ptr<SystemDefinition>,
         std::shared_ptr<NeighborList>, const std::string& >())
        .def("setTuningParam",&T::setTuningParam)
        ;
}
*/

template <class T, class Base> void export_COMPForceComputeGPU(py::module& m, const std::string& name)
{
    pybind::class_<T, std::shared_ptr<T>, Base, std::shared_ptr<Base> >(m, name.c_str(), py::base<ForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setTuningParam",&T::setTuningParam)
        ;
}

#endif // ENABLE_CUDA
