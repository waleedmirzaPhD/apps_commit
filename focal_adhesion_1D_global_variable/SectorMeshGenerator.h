/*
    **************************************************************************************************
    hiperlife - High Performance Library for Finite Elements
    Project homepage: https://git.lacan.upc.edu/HPLFEgroup/hiperlifelib.git
    Copyright (c) 2018 Daniel Santos-Olivan, Alejandro Torres-Sanchez and Guillermo Vilanova
    **************************************************************************************************
    hiperlife is under GNU General Public License ("GPL").
    GNU General Public License ("GPL") copyright permissions statement:
    This file is part of hiperlife, hiperlife is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.
    hiperlife is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PAflinRTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.
    **************************************************************************************************
*/


#ifndef SectorMeshGenerator_H
#define SectorMeshGeneratorH

#include <vector>
#include <functional>
#include <deque>

#include "hl_DistributedData.h"
#include "hl_TypeDefs.h"

#include "hl_MeshCreator.h"
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace hiperlife
{
    class SectorMeshGenerator : public MeshCreator
    {
        public:

            /** @name Constructor / Destructor / Operator
             *   Copy constructors are deleted for memory safety but objects can be created from
             *   temporal objects through move constructors
             *  @{ */
            SectorMeshGenerator(std::string tag = "SectorMeshGenerator", MPI_Comm comm = MPI_COMM_WORLD) : MeshCreator(tag, comm){_meshType= MeshType::Sequential;};
            SectorMeshGenerator(SectorMeshGenerator&& other) = default;
            SectorMeshGenerator(const SectorMeshGenerator& other) = delete;
            ~SectorMeshGenerator() = default;
            SectorMeshGenerator& operator=(const SectorMeshGenerator& other) = delete;
            SectorMeshGenerator& operator=(SectorMeshGenerator&& other) = default;
            /** @} */ // end of constructors


            /** @name Generators for every basis function
             *   Functions to generate meshes valid for every basis function.
             *  @{*/
            void genSector(double h);
            /** @} */ // end of generators


            /** @name Transformations
             *  @{ */
            /** @} */ // end of transformations


      private:

            std::vector<double> _tmp_glob_x_nodes;
            std::vector<int> _tmp_glob_connec;
            std::vector<int> _tmp_glob_creases;
            std::vector<int> _tmp_glob_eflags;

            int _tmp_nPts;
            int _tmp_nElem;
            vtkSmartPointer<vtkPolyData> _tmp_vtkoutput;
            vtkSmartPointer<vtkPolyData> _vtkoutput;

            // Interface functions
            void _saveMesh() override;
            void _reloadMesh() override;
            void _getPRefinedMesh(int orderInc) override;
            void _getPRefinedMesh(int orderIncX, int orderIncY, int orderIncZ) override;
            void _getHRefinedMesh(int refLevel) override;
            void _getBoundaryMesh(std::vector<MAxis> vMAx) override;
            void _genConnecPeriodic();
    };

}
#endif
