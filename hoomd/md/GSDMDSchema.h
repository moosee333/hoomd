
#include "hoomd/extern/gsd.h"
#include "hoomd/GSDDumpWriter.h"
#include "hoomd/GSDReader.h"
#include "hoomd/HOOMDMPI.h"

#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifndef _GSD_MD_Schema_H_
#define _GSD_MD_Schema_H_

struct gsd_schema_md_base
    {
    gsd_schema_md_base(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : m_exec_conf(exec_conf), m_mpi(mpi) {}
    const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    bool m_mpi;
    };

struct gsd_schema_md : public gsd_schema_md_base
    {
    gsd_schema_md(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_md_base(exec_conf, mpi) {}
    int write(gsd_handle& handle, const std::string& name, unsigned int Ntypes, std::vector<Scalar> data, gsd_type type)
        {
        if(!m_exec_conf->isRoot())
            return 0;
        int retval = 0;
        retval |= gsd_write_chunk(&handle, name.c_str(), type, Ntypes, 1, 0, &data);
        return retval;
        }

    bool read(std::shared_ptr<GSDReader> reader, uint64_t frame, const std::string& name, unsigned int Ntypes, std::vector<Scalar> data, gsd_type type)
        {
        bool success = true;
        std::vector<Scalar> d;
        std::cout << "Ntypes is equal to " << Ntypes << "\n";
        if(m_exec_conf->isRoot())
            {
            std::cout << "d is getting resized in the root rank\n";
            d.resize(Ntypes);
            success = reader->readChunk((void *) &d[0], frame, name.c_str(), Ntypes*gsd_sizeof_type(type), Ntypes) && success;
            }

        std::cout << "d is " << d.size() << " long\n";
    #ifdef ENABLE_MPI
        if(m_mpi)
            {
            bcast(d, 0, m_exec_conf->getMPICommunicator()); // broadcast the data
            }
    #endif
        if(!d.size())
            throw std::runtime_error("Error occured while attempting to restore from gsd file.");
        for(unsigned int i = 0; i < Ntypes; i++)
            {
            data[i] = d[i];
            }
        return success;
        }
    };

#endif
