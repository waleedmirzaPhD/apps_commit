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
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.
    **************************************************************************************************
*/


#include <iostream>
#include <array>
#include <random>


#include "hl_TypeDefs.h"
#include "SectorMeshGenerator.h"
#include "hl_Math.h"

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkDelaunay2D.h>
#include <vtkCellLocator.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkFeatureEdges.h>
#include <vtkPointLocator.h>
#include <vtkPolyDataNormals.h>


#include "vtkSurfaceRemeshing.h"


namespace hiperlife
{


    void SectorMeshGenerator::genSector(double h)
    {
        using namespace std;

        _performOrderChecks();

        _nDim = 3;
        _pDim = 2;

        _nVert = 3;
        _nCreases = 2;

        if ( myRank() == 0 )
        {
            double r_in = 0.1;
            double angle = 30.0 * M_PI / 180.0;
            double lenOut = 2.0 * angle;
            double lenIn  = 2.0 * angle * r_in;
            double lenSide = 1.0 - r_in;

            int nIn = int(lenIn/h);
            int nOut = int(lenOut/h);
            int nSide = int(lenSide/h);
            double hIn = 1.0/nIn;
            double hOut = 1.0/nOut;
            double hSide = 1.0/nSide;

            //Create the border of the sector
            vtkSmartPointer<vtkPolyData> vtkoutput = vtkSmartPointer<vtkPolyData>::New();
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            for(int i =0; i < nSide+1; i++)
            {
                points->InsertNextPoint (i * hSide, 0.0, 0.0 );
                // points->InsertNextPoint ( (r_in + i * hSide) * cos(angle), (r_in + i * hSide) * sin(angle), 0.0 );
                points->InsertNextPoint (i * hSide, 1.0, 0.0 );
            }
            for(int i = 1; i < nIn; i++)
                // points->InsertNextPoint ( r_in * cos(i * hIn), r_in * sin(i * hIn), 0.0 );
                points->InsertNextPoint(0.0, i * hIn, 0.0 );

            for(int i = 1; i < nOut; i++)
                points->InsertNextPoint(1.0, i * hOut, 0.0 );

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<> dist(0, 1);
            for(int i = 0; i < 100; i++)
            {
                double r = dist(e2);
                double a = dist(e2);
                points->InsertNextPoint ( 0.5+(1.-h)*(r-0.5), 0.5+(1.-h)*(a-0.5), 0.0 );
            }
            vtkoutput->SetPoints(points);

            vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
            delaunay->SetInputData(vtkoutput);
            delaunay->Update();
            vtkoutput = delaunay->GetOutput();

            // vtkoutput->SetPoints(points);
            // writer = vtkSmartPointer<vtkPolyDataWriter>::New();
            // writer->SetInputData(vtkoutput);
            // writer->SetFileName("delaunay.vtk");
            // writer->Update();

            points = vtkSmartPointer<vtkPoints>::New();
            for(int i = 0; i < vtkoutput->GetNumberOfPoints(); i++)
            {
                double* x = vtkoutput->GetPoints()->GetPoint(i);
                points->InsertNextPoint((r_in + (1.0-r_in) * x[0]) * cos(x[1] * angle), (r_in + (1.0-r_in) * x[0]) * sin(x[1] * angle), 0.0);
            }
            vtkoutput->SetPoints(points);


            // Remesh the sector
            vtkSmartPointer<vtkSurfaceRemeshing> vtkRemeshing = vtkSurfaceRemeshing::New();
            vtkRemeshing->SetInputData( vtkoutput );
            vtkRemeshing->SetElementSizeModeToTargetArea();
            vtkRemeshing->SetTargetArea(h*h/2.0);
            vtkRemeshing->SetNumberOfIterations(10);
            vtkRemeshing->SetTriangleSplitFactor(10.0);
            vtkRemeshing->SetNumberOfConnectivityOptimizationIterations(100);
            vtkRemeshing->PreserveBoundaryEdgesOn(); //This will ensure the edges are preserved
            vtkRemeshing->Update();
            vtkoutput = vtkRemeshing->GetOutput();

            vtkSmartPointer<vtkCleanPolyData> vtkClean = vtkCleanPolyData::New();
            vtkClean->SetInputData(vtkoutput);
            vtkClean->Update();
            vtkoutput = vtkClean->GetOutput();

            // points = vtkSmartPointer<vtkPoints>::New();
            // for(int i = 0; i < vtkoutput->GetNumberOfPoints(); i++)
            // {
            //     double* x = vtkoutput->GetPoints()->GetPoint(i);
            //     points->InsertNextPoint((sqrt(x[0]*x[0]+x[1]*x[1])-r_in)/(1.0-r_in), atan2(x[1], x[0])/angle, 0.0);
            // }
            // vtkoutput->SetPoints(points);

            vtkoutput->BuildCells();
            vtkoutput->BuildLinks();


            points = vtkSmartPointer<vtkPoints>::New();
            for(int i = 0; i < vtkoutput->GetNumberOfPoints(); i++)
                points->InsertNextPoint(vtkoutput->GetPoint(i));

            //Add the last points
            vtkSmartPointer<vtkCellLocator> cellLocator = vtkCellLocator::New();
            cellLocator->SetDataSet(vtkoutput);
            cellLocator->BuildLocator();
            int n{};

            vector<int> removedCells;
            vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
            for ( int i = 0; i <=1 ; i++ )
            {
                for ( int j = 0; j <= 1; j++ )
                {
                    // double x[3] = {double(i), double(j), 0.0};
                    double x[3] = {(r_in + (1.0-r_in) * i) * cos(j * angle), (r_in + (1.0-r_in) * i) * sin(j * angle), 0.0};

                    double cpoint[3];
                    vtkIdType cellId;
                    int subId;
                    double dist;
                    cellLocator->FindClosestPoint(x, cpoint, cellId, subId, dist);

                    if(dist > 1.E-8)
                    {
                        vtkSmartPointer<vtkCell> cell = vtkoutput->GetCell(cellId);
                        vtkSmartPointer<vtkIdList> cellPoints = cell->GetPointIds();

                        int i1 = cellPoints->GetId(0);
                        int i2 = cellPoints->GetId(1);
                        int i3 = cellPoints->GetId(2);

                        vector<pair<int,int>> edges = {{i1,i2},{i2,i3},{i3,i1}};

                        points->InsertNextPoint(x);

                        double area = 0.0;

                        pair<int,int> edgeIds;
                        for(auto& e: edges)
                        {
                            vtkSmartPointer<vtkIdList> nborsIds = vtkIdList::New();
                            vtkSmartPointer<vtkIdList> edge = vtkIdList::New();
                            edge->InsertNextId(e.first);
                            edge->InsertNextId(e.second);
                            
                            vtkoutput->GetCellNeighbors(cellId, edge, nborsIds);

                            if(nborsIds->GetNumberOfIds() == 0)
                            {
                                double p1[3]; 
                                vtkoutput->GetPoints()->GetPoint(e.first, p1);
                                double p2[3];
                                vtkoutput->GetPoints()->GetPoint(e.second, p2);
                                
                                double area_new = vtkTriangle::TriangleArea(x, p1, p2);
                                if(area_new > area)
                                {
                                    area = area_new;
                                    edgeIds.first = e.first;
                                    edgeIds.second = e.second;
                                }
                            }
                        }
                        if(edgeIds.first != 0 and edgeIds.second != 0)
                        {
                            double p1[3]; 
                            vtkoutput->GetPoints()->GetPoint(edgeIds.first, p1);
                            double p2[3];
                            vtkoutput->GetPoints()->GetPoint(edgeIds.second, p2);
                            int opposite = edgeIds.first == i1 ? (edgeIds.second == i2 ? i3 : i2) : (edgeIds.first == i2 ? (edgeIds.second == i1 ? i3: i1) : (edgeIds.second == i1 ? i2: i1));
                            // double p3[3];
                            // vtkoutput->GetPoints()->GetPoint(p3, opposite);

                            double pmid[3] = {0.5*(p1[0]+p2[0]),
                                              0.5*(p1[1]+p2[1]),
                                              0.5*(p1[2]+p2[2])};
                    
                            points->InsertNextPoint(pmid);

                            vtkSmartPointer<vtkIdList> triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(edgeIds.first);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+1);
                            triangle->InsertNextId(opposite);
                            cells->InsertNextCell(triangle);

                            triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(edgeIds.first);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+0);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+1);
                            cells->InsertNextCell(triangle);

                            triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(edgeIds.second);
                            triangle->InsertNextId(opposite);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+1);
                            cells->InsertNextCell(triangle);

                            triangle = vtkSmartPointer<vtkIdList>::New();
                            triangle->InsertNextId(edgeIds.second);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+1);
                            triangle->InsertNextId(vtkoutput->GetNumberOfPoints()+2*n+0);
                            cells->InsertNextCell(triangle);

                            removedCells.push_back(cellId);
                            n++;
                        }
                        else
                            cout << "PROBLEMA" << endl;
                    }
                    
                }
            }

            for(int i =0; i < vtkoutput->GetNumberOfCells(); i++)
            {
                if(find(removedCells.begin(), removedCells.end(), i) == removedCells.end())
                    cells->InsertNextCell(vtkoutput->GetCell(i));
            }
            
            vtkoutput = vtkSmartPointer<vtkPolyData>::New();
            vtkoutput->SetPoints(points);
            vtkoutput->SetPolys(cells);


            vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
            normals->SetInputData(vtkoutput);
            normals->ConsistencyOn();
            normals->ComputePointNormalsOn();
            normals->Update();

            if(normals->GetOutput()->GetPointData()->GetArray("Normals")->GetTuple(0)[2] < 0)
            {
                normals->FlipNormalsOn();
                normals->Update();
            }

            vtkoutput = normals->GetOutput();

            _vtkoutput = vtkoutput;

            _nPts  = vtkoutput->GetNumberOfPoints();
            _nElem = vtkoutput->GetNumberOfCells();

            _glob_x_nodes.resize(_nPts*_nDim,0.0);
            for( int i = 0; i < _nPts; i++ )
            {
                double x[3];
                vtkoutput->GetPoint(i, x);
                _glob_x_nodes[_nDim*i+0] = x[0];
                _glob_x_nodes[_nDim*i+1] = x[1];
                _glob_x_nodes[_nDim*i+2] = x[2];
            }

            _glob_connec.resize(_nElem*_nVert,0);
            for( int i = 0; i < _nElem; i++ )
            {
                vtkSmartPointer<vtkIdList> ptsIds = vtkIdList::New();
                vtkoutput->GetCellPoints(i, ptsIds);
                _glob_connec[_nVert*i+0] = ptsIds->GetId(0);
                _glob_connec[_nVert*i+1] = ptsIds->GetId(1);
                _glob_connec[_nVert*i+2] = ptsIds->GetId(2);

            }

            _glob_creases.resize(_nCreases*_nPts, 0);
            _glob_eflags.resize(_nElem*_nEFlags,0);
        }
        else
        {
            _glob_x_nodes.resize(_nDim, 0);
            _glob_connec.resize(_nVert, 0);
            _glob_creases.resize(_nCreases, 0);
            _glob_eflags.resize(_nEFlags, 0);
        }
        _genConnecPeriodic();    

        //Communicate number of points and elements
        MPI_Barrier(comm());
        MPI_Bcast(&_nPts,  1, MPI_INT, 0, comm());
        MPI_Bcast(&_nElem, 1, MPI_INT, 0, comm());
    }

    //Check, save and reload mesh
    void SectorMeshGenerator::_saveMesh()
    {

        _tmp_glob_connec  = _glob_connec;
        _tmp_glob_x_nodes = _glob_x_nodes;
        _tmp_glob_creases = _glob_creases;
        _tmp_glob_eflags  = _glob_eflags;

        _tmp_nElem = _nElem;
        _tmp_nPts = _nPts;

        _tmp_vtkoutput = _vtkoutput;
    }

    void SectorMeshGenerator::_reloadMesh()
    {

        _glob_connec = _tmp_glob_connec ;
        _glob_x_nodes = _tmp_glob_x_nodes;
        _glob_creases = _tmp_glob_creases;
        _glob_eflags = _tmp_glob_eflags;

        _nElem = _tmp_nElem;
        _nPts = _tmp_nPts;

        _vtkoutput = _tmp_vtkoutput;
    }


    //p-Refined mesh
    void SectorMeshGenerator::_getPRefinedMesh(int orderInc)
    {
        (void) orderInc;
        using namespace std;

        (void) orderInc;

        //FIXME: make it work for squares and higher-order elements
        if(_eType!=ElemType::Triang or _bfType != BasisFuncType::Linear)
                throw runtime_error("MeshLoader::getPRefinedMesh: Not implemented for elem type different than triangles or basis functions different from linear.");

        _bfType = BasisFuncType::Lagrangian;
        setBasisFuncOrder(2);
        _performOrderChecks();

        _nVert = 6;
        _pRefinedMesh = true;

        _loc_nPts=0;
        _loc_nElem=0;

        if (_myRank == 0)
        {
            vector<int> glob_connec(_nElem*_nVert);

            vector<pair<int,int>> edges;
            for(int e = 0; e < _nElem;  e++)
            {
                /*
                   2
                   | \
                   |   \
                   |     \
                   0------1 //local labels
                */

                //global labels
                int e0 = _glob_connec[e*3+0];  //_glob_connec is matrix of size _nElem * 3
                int e1 = _glob_connec[e*3+1];
                int e2 = _glob_connec[e*3+2];

                edges.push_back({min(e0,e1),max(e0,e1)});
                edges.push_back({min(e1,e2),max(e1,e2)});
                edges.push_back({min(e0,e2),max(e0,e2)});
            }

            sort(edges.begin(),edges.end());

            edges.erase(unique(edges.begin(), edges.end()), edges.end());

            int nPts = _nPts + edges.size();

            vector<int> newlabel_new(edges.size(),-1);
            vector<int> newlabel_old(_nPts,-1);

            vector<double> glob_x_nodes;

            int lastLabel{};
            for(int e = 0; e < _nElem;  e++)
            {
                /*
                   2
                   | \
                   |   \
                   |     \
                   0------1 //local labels
                */

                //global labels
                int e0 = _glob_connec[e*3+0];  //_glob_connec is matrix of size _nElem * 3
                int e1 = _glob_connec[e*3+1];
                int e2 = _glob_connec[e*3+2];

                double* x0 = &_glob_x_nodes[e0*_nDim]; //positions of the nodes
                double* x1 = &_glob_x_nodes[e1*_nDim];
                double* x2 = &_glob_x_nodes[e2*_nDim];


                //FIXME: make this more efficient
                int l0 = find(edges.begin(),edges.end(),std::pair<int,int>({min(e0,e1),max(e0,e1)}))-edges.begin();
                int l1 = find(edges.begin(),edges.end(),std::pair<int,int>({min(e1,e2),max(e1,e2)}))-edges.begin();
                int l2 = find(edges.begin(),edges.end(),std::pair<int,int>({min(e0,e2),max(e0,e2)}))-edges.begin();

                /*
                   5
                   | \
                   3   4
                   |     \
                   0---1---2
                */

                if(newlabel_old[e0] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(x0[m]);

                    newlabel_old[e0] = lastLabel;
                    glob_connec[e*6+0] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+0] = newlabel_old[e0];

                if(newlabel_new[l0] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(0.5*(x0[m]+x1[m]));

                    newlabel_new[l0] = lastLabel;
                    glob_connec[e*6+1] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+1] = newlabel_new[l0];

                if(newlabel_old[e1] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(x1[m]);

                    newlabel_old[e1] = lastLabel;
                    glob_connec[e*6+2] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+2] = newlabel_old[e1];

                if(newlabel_new[l2] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(0.5*(x0[m]+x2[m]));

                    newlabel_new[l2] = lastLabel;
                    glob_connec[e*6+3] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+3] = newlabel_new[l2];

                if(newlabel_new[l1] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(0.5*(x1[m]+x2[m]));

                    newlabel_new[l1] = lastLabel;
                    glob_connec[e*6+4] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+4] = newlabel_new[l1];


                if(newlabel_old[e2] == -1)
                {
                    for(int m = 0; m< _nDim; m++)
                        glob_x_nodes.push_back(x2[m]);

                    newlabel_old[e2] = lastLabel;
                    glob_connec[e*6+5] = lastLabel;
                    lastLabel++;
                }
                else
                    glob_connec[e*6+5] = newlabel_old[e2];
            }

            _nPts = nPts;
            _glob_x_nodes = glob_x_nodes;
            _glob_connec  = glob_connec;

            _glob_creases.resize(_nPts*_nCreases,0);
            _glob_eflags.resize(_nElem*_nEFlags,0);
        }
        else
        {
            _glob_x_nodes.resize(_nDim, 0);
            _glob_connec.resize(_nVert, 0);
            _glob_creases.resize(_nCreases, 0);
            _glob_eflags.resize(_nEFlags, 0);
        }

        _genConnecPeriodic();    

        //Communicate number of points and elements
        MPI_Barrier(comm());
        MPI_Bcast(&_nPts,  1, MPI_INT, 0, comm());
        MPI_Bcast(&_nElem, 1, MPI_INT, 0, comm());
    }

    void SectorMeshGenerator::_getPRefinedMesh(int orderIncX, int orderIncY, int orderIncZ)
    {
        (void) orderIncX;
        (void) orderIncY;
        (void) orderIncZ;
    }

    //h-Refined mesh
    void SectorMeshGenerator::_getHRefinedMesh(int refLevel)
    {

        if ( myRank() == 0 )
        {
            vtkSmartPointer<vtkLinearSubdivisionFilter>  subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
            subdivisionFilter->SetInputData(_vtkoutput);
            subdivisionFilter->SetNumberOfSubdivisions(refLevel);
            subdivisionFilter->Update();

            _vtkoutput = subdivisionFilter->GetOutput();
            
            _nPts  = _vtkoutput->GetNumberOfPoints();
            _nElem = _vtkoutput->GetNumberOfCells();


            _glob_x_nodes.resize(_nPts*_nDim,0.0);
            for( int i = 0; i < _nPts; i++ )
            {
                double x[3];
                _vtkoutput->GetPoint(i, x);

                _glob_x_nodes[_nDim*i+0] = x[0];
                _glob_x_nodes[_nDim*i+1] = x[1];
                _glob_x_nodes[_nDim*i+2] = x[2];
            }

            _glob_connec.resize(_nElem*_nVert,0);
            for( int i = 0; i < _nElem; i++ )
            {
                vtkSmartPointer<vtkIdList> ptsIds = vtkIdList::New();
                _vtkoutput->GetCellPoints(i, ptsIds);
                _glob_connec[_nVert*i+0] = ptsIds->GetId(0);
                _glob_connec[_nVert*i+1] = ptsIds->GetId(1);
                _glob_connec[_nVert*i+2] = ptsIds->GetId(2);
            }

            _glob_creases.resize(_nPts*_nCreases,0);
            
            
            vtkSmartPointer<vtkFeatureEdges> edges = vtkSmartPointer<vtkFeatureEdges>::New();
            edges->SetInputData(_vtkoutput);
            edges->BoundaryEdgesOn();
            edges->FeatureEdgesOff();
            edges->NonManifoldEdgesOff();
            edges->ManifoldEdgesOff();
            edges->Update();
            
            auto polydata_b = edges->GetOutput();
            
            vtkSmartPointer<vtkPointLocator> loc = vtkSmartPointer<vtkPointLocator>::New();
            loc->SetDataSet(_vtkoutput);
            loc->BuildLocator();
            
            for(int i = 0; i < polydata_b->GetPoints()->GetNumberOfPoints(); i++)
            {
                std::array<double,3> x;
                polydata_b->GetPoints()->GetPoint(i, x.data() );
                int j = loc->FindClosestPoint(x.data());

                _glob_creases[_nCreases*j] = 1;

            }

            _glob_eflags.resize(_nElem*_nEFlags,0);
        }
        else
        {
            _glob_x_nodes.resize(_nDim, 0);
            _glob_connec.resize(_nVert, 0);
            _glob_creases.resize(_nCreases, 0);
            _glob_eflags.resize(_nEFlags, 0);
        }

        _genConnecPeriodic();    

        //Communicate number of points and elements
        MPI_Barrier(comm());
        MPI_Bcast(&_nPts,  1, MPI_INT, 0, comm());
        MPI_Bcast(&_nElem, 1, MPI_INT, 0, comm());
    }



    //Boundary mesh
    void SectorMeshGenerator::_getBoundaryMesh(std::vector<MAxis> vMAx)
    {
        (void) vMAx;
    }

    void SectorMeshGenerator::_genConnecPeriodic()
    {
        double angle = 30.0 * M_PI / 180.0;

        _linearConstraints.clear();
        if(_myRank == 0)
        {
            auto points = vtkSmartPointer<vtkPoints>::New();
            for(int i = 0; i < _nPts; i++)
            {
                points->InsertNextPoint(_glob_x_nodes[i*_nDim+0], _glob_x_nodes[i*_nDim+1], _glob_x_nodes[i*_nDim+2]);
            }
            vtkSmartPointer<vtkPolyData> vtkoutput = vtkSmartPointer<vtkPolyData>::New();
            vtkoutput->SetPoints(points);

            vtkSmartPointer<vtkPointLocator> loc = vtkSmartPointer<vtkPointLocator>::New();
            loc->SetDataSet(vtkoutput);
            loc->BuildLocator();

            for(int i = 0; i < _nPts; i++)
            {
                if(_glob_x_nodes[_nDim*i+1] < 1.E-8)
                {
                    std::array<double,3> x = {_glob_x_nodes[_nDim*i+0]*cos(angle), _glob_x_nodes[_nDim*i+0]*sin(angle), _glob_x_nodes[_nDim*i+2]};
                    int j = loc->FindClosestPoint(x.data());
                    // _connectPeriodic.push_back({j, i});
                    _linearConstraints.push_back({{j}, {i}});
                }

                bool crease = false;
                _glob_creases[_nCreases*i+0] = 0;
                _glob_creases[_nCreases*i+1] = 1;
                if(sqrt(_glob_x_nodes[_nDim*i+0]*_glob_x_nodes[_nDim*i+0]+_glob_x_nodes[_nDim*i+1]*_glob_x_nodes[_nDim*i+1]) < 0.1+1.E-4)
                {
                    _glob_creases[_nCreases*i+0] += 1;
                    _glob_creases[_nCreases*i+1] *= 2;
                    crease = true;
                }
                if(sqrt(_glob_x_nodes[_nDim*i+0]*_glob_x_nodes[_nDim*i+0]+_glob_x_nodes[_nDim*i+1]*_glob_x_nodes[_nDim*i+1]) > 1.0-1.E-4)
                {
                    _glob_creases[_nCreases*i+0] += 1;
                    _glob_creases[_nCreases*i+1] *= 3;
                    crease = true;
                }
                if (_glob_x_nodes[_nDim*i+1] < 1.E-8)
                {
                    _glob_creases[_nCreases*i+0] += 1;
                    _glob_creases[_nCreases*i+1] *= 5;
                    crease = true;
                }
                if (abs(_glob_x_nodes[_nDim*i+1]/_glob_x_nodes[_nDim*i+0] - tan(30.0*M_PI/180.0)) < 1.E-4)
                {
                    _glob_creases[_nCreases*i+0] += 1;
                    _glob_creases[_nCreases*i+1] *= 7;
                    crease = true;
                }
                if (!crease)
                {
                    _glob_creases[_nCreases*i+0] = 0;
                    _glob_creases[_nCreases*i+1] = -1;
                }
            }
        }
        cout << "Number of pairs:" << _linearConstraints.size() << endl;
    }


}
