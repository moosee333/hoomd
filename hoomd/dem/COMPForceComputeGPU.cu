
scalar4_tex_t pdata_quat_tex;

template<typename Potential, unsigned int shift_mode, unsigned int compute_virial>
__global__ void compute_comp_force_gpu_kernel(Scalar4 *d_force,
                                              Scalar4 *d_quat,
                                              Scalar4 *d_torque,
                                              Scalar *d_virial,
                                              const unsigned int virial_pitch,
                                              const unsigned int N,
                                              const Scalar4 *d_pos,
                                              const Scalar *d_diameter,
                                              const Scalar *d_charge,
                                              const BoxDim box,
                                              const unsigned int *d_n_neigh,
                                              const unsigned int *d_nlist,
                                              const unsigned int *d_head_list,
                                              const typename Potential::param_type *d_params,
                                              const Scalar *d_rcutsq,
                                              const Scalar *d_ronsq,
                                              const Scalar3 *d_verts,
                                              const unsigned int *d_typeVertStart,
                                              const unsigned int *d_typeVertCount,
                                              const unsigned int ntypes,
                                              const unsigned int tpp,
                                              const unsigned int nverts)
{
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    extern __shared__ char s_data[];
    unsigned int shmOffset(0);
    typename Potential::param_type *s_params =
        (typename Potential::param_type *)(&s_data[shmOffset]);
    shmOffset += num_typ_parameters*sizeof(Potential::param_type);

    // align to size of Scalar3
    if(shmOffset % sizeof(Scalar3))
       shmOffset += sizeof(Scalar3) - shmOffset % sizeof(Scalar3);
    Scalar3 *s_verts((Scalar3*) &s_data[shmOffset]);
    shmOffset += nverts*sizeof(Scalar3);

    Scalar *s_rcutsq = (Scalar *)(&s_data[shmOffset]);
    shmOffset += num_typ_parameters*sizeof(Scalar);
    Scalar *s_ronsq = (Scalar *)(&s_data[shmOffset]);
    if(shift_mode == 2)
        shmOffset += num_typ_parameters*sizeof(Scalar);
    unsigned int *s_typeVertStart((unsigned int*) &s_data[shmOffset]);
    shmOffset += ntypes*sizeof(unsigned int);
    unsigned int *s_typeVertCount((unsigned int*) &s_data[shmOffset]);

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
    {
        if (cur_offset + threadIdx.x < num_typ_parameters)
        {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            if (shift_mode == 2)
                s_ronsq[cur_offset + threadIdx.x] = d_ronsq[cur_offset + threadIdx.x];
        }
    }

    // load in the per type parameters
    for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
    {
        if (cur_offset + threadIdx.x < ntypes)
        {
            s_typeVertStart[cur_offset + threadIdx.x] = d_typeVertStart[cur_offset + threadIdx.x];
            s_typeVertCount[cur_offset + threadIdx.x] = d_typeVertCount[cur_offset + threadIdx.x];
        }
    }

    // load in the per vertex parameters
    for (unsigned int cur_offset = 0; cur_offset < nverts; cur_offset += blockDim.x)
    {
        if (cur_offset + threadIdx.x < nverts)
            s_verts[cur_offset + threadIdx.x] = d_verts[cur_offset + threadIdx.x];
    }

    __syncthreads();

    // start by identifying which particle we are to handle
    unsigned int idx;
    if (gridDim.y > 1)
    {
        // if we have blocks in the y-direction, the fermi-workaround is in place
        idx = (blockIdx.x + blockIdx.y * 65535) * (blockDim.x/tpp) + threadIdx.x/tpp;
    }
    else
    {
        idx = blockIdx.x * (blockDim.x/tpp) + threadIdx.x/tpp;
    }

    bool active = true;

    if (idx >= N)
    {
        // need to mask this thread, but still participate in warp-level reduction (because of __syncthreads())
        active = false;
    }

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar3 torque = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virialxx = Scalar(0.0);
    Scalar virialxy = Scalar(0.0);
    Scalar virialxz = Scalar(0.0);
    Scalar virialyy = Scalar(0.0);
    Scalar virialyz = Scalar(0.0);
    Scalar virialzz = Scalar(0.0);

    if (active)
    {
        // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
        unsigned int n_neigh = d_n_neigh[idx];

        // read in the position of our particle.
        // (MEM TRANSFER: 16 bytes)
        Scalar4 postypei = texFetchScalar4(d_pos, pdata_pos_tex, idx);
        const vec3<Scalar> ri(postypei.x, postypei.y, postypei.z);
        const unsigned int typei = __scalar_as_int(postypei.w);
        const quat<Scalar> quati(texFetchScalar4(d_quat, pdata_quat_tex, idx));

        Scalar di;
        if (Potential::needsDiameter())
            di = texFetchScalar(d_diameter, pdata_diam_tex, idx);
        else
            di += Scalar(1.0); // shutup compiler warning
        Scalar qi;
        if (Potential::needsCharge())
            qi = texFetchScalar(d_charge, pdata_charge_tex, idx);
        else
            qi += Scalar(1.0); // shutup compiler warning

        unsigned int my_head = d_head_list[idx];
        unsigned int cur_j = 0;

        unsigned int next_j((threadIdx.x%tpp < n_neigh) ? d_nlist[my_head + threadIdx.x%tpp] : 0);
        // loop over neighbors
        // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
        // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
        // inside the loop
            for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
            {
                {
                    // read the current neighbor index (MEM TRANSFER: 4 bytes)
                    cur_j = next_j;
                    if (neigh_idx+tpp < n_neigh)
                        next_j = d_nlist[my_head + neigh_idx + tpp];

                    // get the neighbor's position (MEM TRANSFER: 16 bytes)
                    Scalar4 postypej = texFetchScalar4(d_pos, pdata_pos_tex, cur_j);
                    const vec3<Scalar> rj(postypej.x, postypej.y, postypej.z);
                    const unsigned int typej = __scalar_as_int(postypej.w);
                    const quat<Scalar> quatj(texFetchScalar4(d_quat, pdata_quat_tex, cur_j));

                    Scalar dj = Scalar(0.0);
                    if (Potential::needsDiameter())
                        dj = texFetchScalar(d_diameter, pdata_diam_tex, cur_j);
                    else
                        dj += Scalar(1.0); // shutup compiler warning

                    Scalar qj = Scalar(0.0);
                    if (Potential::needsCharge())
                        qj = texFetchScalar(d_charge, pdata_charge_tex, cur_j);
                    else
                        qj += Scalar(1.0); // shutup compiler warning

                    // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                    vec3<Scalar> rij(rj - ri);

                    // apply periodic boundary conditions: (FLOPS 12)
                    const Scalar3 image(box.minImage(vec_to_scalar3(rij)));
                    rij = vec3<Scalar>(image.x, image.y, image.z);

                    // calculate r squard (FLOPS: 5)
                    Scalar rsq = dot(rij, rij);

                    // access the per type pair parameters
                    unsigned int typpair = typpair_idx(typei, typej);
                    Scalar rcutsq = s_rcutsq[typpair];
                    typename Potential::param_type param = s_params[typpair];
                    Scalar ronsq = Scalar(0.0);
                    if (shift_mode == 2)
                        ronsq = s_ronsq[typpair];

                    // design specifies that energies are shifted if
                    // 1) shift mode is set to shift
                    // or 2) shift mode is explor and ron > rcut
                    bool energy_shift = false;
                    if (shift_mode == 1)
                        energy_shift = true;
                    else if (shift_mode == 2)
                    {
                        if (ronsq > rcutsq)
                            energy_shift = true;
                    }

                    // evaluate the potential
                    vec3<Scalar> forceij(0, 0, 0);
                    vec3<Scalar> torqueij(0, 0, 0);
                    Scalar energyij(0);

            for(unsigned int vi(0); vi < s_typeVertCount[typei]; ++vi)
            {
                const vec3<Scalar> verti(rotate(quati,
                    vec3<Scalar>(s_verts[s_typeVertStart[typei] + vi])));
                for(unsigned int vj(0); vj < s_typeVertCount[typej]; ++vj)
                {
                    const vec3<Scalar> vertj(rotate(quatj,
                        vec3<Scalar>(s_verts[s_typeVertStart[typej] + vj])));
                    const vec3<Scalar> rvivj(rij + vertj - verti);
                    const Scalar rsq(dot(rvivj, rvivj));

                    Potential eval(rsq, rcutsq, param);
                    Scalar force_divr(0);
                    Scalar pairEnergy(0);
                    if(Potential::needsDiameter())
                        eval.setDiameter(di, dj);
                    if(Potential::needsCharge())
                        eval.setCharge(qi, qj);

                    bool evaluated = eval.evalForceAndEnergy(force_divr, pairEnergy, energy_shift);

                    if(evaluated)
                    {
                        const vec3<Scalar> force(-force_divr*rvivj);
                        forceij += force;
                        energyij += pairEnergy;
                        torqueij += cross(verti, force);
                    }
                }
            }

            force.x += forceij.x;
            force.y += forceij.y;
            force.z += forceij.z;
            force.w += energyij;

            torque.x += torqueij.x;
            torque.y += torqueij.y;
            torque.z += torqueij.z;

                    // calculate the virial
                    if (compute_virial)
                    {
                        virialxx -= forceij.x*rij.x*Scalar(0.5);
                        virialxy -= forceij.x*rij.y*Scalar(0.5);
                        virialxz -= forceij.x*rij.z*Scalar(0.5);
                        virialyy -= forceij.y*rij.y*Scalar(0.5);
                        virialyz -= forceij.y*rij.z*Scalar(0.5);
                        virialzz -= forceij.z*rij.z*Scalar(0.5);
                    }

                }
            }

        // potential energy per particle must be halved
        force.w *= Scalar(0.5);
    }

    // we need to access a separate portion of shared memory to avoid race conditions
    const unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename Potential::param_type)) * typpair_idx.getNumElements() +
        2*ntypes*sizeof(unsigned int) + nverts*sizeof(Scalar3) +
        // Add in padding for alignment of Scalar3 if necessary
        (num_typ_parameters*sizeof(Potential::param_type) % sizeof(Scalar3) != 0)*
        (sizeof(Scalar3) - (num_typ_parameters*sizeof(Potential::param_type) % sizeof(Scalar3)));

    // need to declare as volatile, because we are using warp-synchronous programming
    volatile Scalar *sh = (Scalar *) &s_data[shared_bytes];

    unsigned int cta_offs = (threadIdx.x/tpp)*tpp;

    // reduce force over threads in cta
    force.x = warp_reduce(tpp, threadIdx.x % tpp, force.x, &sh[cta_offs]);
    force.y = warp_reduce(tpp, threadIdx.x % tpp, force.y, &sh[cta_offs]);
    force.z = warp_reduce(tpp, threadIdx.x % tpp, force.z, &sh[cta_offs]);
    force.w = warp_reduce(tpp, threadIdx.x % tpp, force.w, &sh[cta_offs]);

    torque.x = warp_reduce(tpp, threadIdx.x % tpp, torque.x, &sh[cta_offs]);
    torque.y = warp_reduce(tpp, threadIdx.x % tpp, torque.y, &sh[cta_offs]);
    torque.z = warp_reduce(tpp, threadIdx.x % tpp, torque.z, &sh[cta_offs]);

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    if (active && threadIdx.x % tpp == 0)
    {
        d_force[idx] = force;
        d_torque[idx] = make_scalar4(torque.x, torque.y, torque.z, Scalar(0.0));
    }

    if (compute_virial)
    {
        virialxx = warp_reduce(tpp, threadIdx.x % tpp, virialxx, &sh[cta_offs]);
        virialxy = warp_reduce(tpp, threadIdx.x % tpp, virialxy, &sh[cta_offs]);
        virialxz = warp_reduce(tpp, threadIdx.x % tpp, virialxz, &sh[cta_offs]);
        virialyy = warp_reduce(tpp, threadIdx.x % tpp, virialyy, &sh[cta_offs]);
        virialyz = warp_reduce(tpp, threadIdx.x % tpp, virialyz, &sh[cta_offs]);
        virialzz = warp_reduce(tpp, threadIdx.x % tpp, virialzz, &sh[cta_offs]);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x %tpp == 0)
        {
            d_virial[0*virial_pitch+idx] = virialxx;
            d_virial[1*virial_pitch+idx] = virialxy;
            d_virial[2*virial_pitch+idx] = virialxz;
            d_virial[3*virial_pitch+idx] = virialyy;
            d_virial[4*virial_pitch+idx] = virialyz;
            d_virial[5*virial_pitch+idx] = virialzz;
        }
    }
}

template<typename Potential>
cudaError_t compute_comp_force_gpu(pair_args_t pair_args, Scalar4 *d_quat,
                                   Scalar4 *d_torque,
                                   const GeometryArgs &geom_args,
                                   typename Potential::param_type *d_params)
{
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    // threads per particle

    // setup the grid to run the kernel
    unsigned int block_size = pair_args.block_size;
    unsigned int tpp = pair_args.threads_per_particle;

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename Potential::param_type))
        * typpair_idx.getNumElements() + 2*pair_args.ntypes*sizeof(unsigned int) +
        sizeof(Scalar3)*geom_args.numVertices;

    // Add in padding for alignment of Scalar3 if necessary
    if(typpair_idx.getNumElements()*sizeof(typename Potential::param_type) % sizeof(Scalar3))
        shared_bytes += sizeof(Scalar3) -
            typpair_idx.getNumElements()*sizeof(typename Potential::param_type) % sizeof(Scalar3);

    if(pair_args.compute_capability < 35)
    {
        pdata_quat_tex.normalized = false;
        pdata_quat_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, pdata_quat_tex, d_quat, sizeof(Scalar4)*pair_args.n_max);
    }

    // Launch kernel
    if (pair_args.compute_virial)
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 0, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 0, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 1, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 1, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            case 2:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 2, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 2, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            default:
                break;
            }
        }
    else
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 0, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 0, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 1, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 1, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            case 2:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(compute_comp_force_gpu_kernel<Potential, 2, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                if (pair_args.compute_capability < 30)
                    {
                    shared_bytes += sizeof(Scalar)*block_size;
                    }

                compute_comp_force_gpu_kernel<Potential, 2, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, d_quat, d_torque, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, geom_args.vertices, geom_args.typeStarts, geom_args.typeCounts, pair_args.ntypes,
                  tpp, geom_args.numVertices);
                break;
                }
            default:
                break;
            }
        }

    return cudaSuccess;
}
