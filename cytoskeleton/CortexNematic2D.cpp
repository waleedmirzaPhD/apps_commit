
/// C++ headers
#include <iostream>
#include <mpi.h>
#include <fstream>
/// Trilinos headers
#include <Teuchos_RCP.hpp>
/// hiperlife headers
#include "hl_TypeDefs.h"
#include "hl_Geometry.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_Tensor.h"
#include "hl_MeshLoader.h"
#include <Teuchos_CommandLineProcessor.hpp>
#include "hl_ConfigFile.h"
#include "hl_ConsistencyCheck.h"
#include "AuxCortexNematic2D.h"
#include "hl_LinearSolver_Direct_Amesos.h"
#include "hl_LinearSolver_Iterative_AztecOO.h"
#include "hl_NonlinearSolver_NewtonRaphson.h"
#include "hl_LinearSolver_Direct_MUMPS.h"
#include <random>
#include <cmath>


int main ( int argc, char *argv[]  )
{
    using namespace std;
    using namespace hiperlife;
    using Teuchos::rcp;
    using Teuchos::RCP;


    //Initialize MPI
    MPI_Init(nullptr, nullptr);

    bool printVtk{true};
    bool printFile{true};
    bool diagnosticFlag{false};

    const char *config_filename = argv[1];
    ConfigFile config(config_filename);

    int restart{};
    config.readInto(restart, "restart");

    // path to store the .csv file for output
    string csvPathDiss = "globalIntegralsDiss.csv" ;
    config.readInto(csvPathDiss, "csvPathDiss");

    //string csvPathEnergy = "globalIntegralsEnergy.csv" ;
    //config.readInto(csvPathEnergy, "csvPathEnergy");


    //Number of Gauss points
    int gPts{};
    config.readInto(gPts, "gPts");

    // Geoemtric parameters for structured mesh
    double r_in = 3.0;
    config.readInto(r_in, "r_in");

    double r_out = 5.0;
    config.readInto(r_out, "r_out");

    double phi = M_PI/2.0;
    config.readInto(phi, "phi");

    int nx=100;
    config.readInto(nx, "nx");

    int ny=20;
    config.readInto(ny, "ny");

    //Model parameters
    double visc      = 20.0;
    config.readInto(visc, "visc");

    double cvisc     = 3.0;
    config.readInto(cvisc, "cvisc");

    double rvisc = 1.0;
    config.readInto(rvisc, "rvisc");

    double frank   = 1.0;
    config.readInto(frank, "frank");

    double lambda_iso   = 0.0;//10.0;  // Contractile case
    config.readInto(lambda_iso, "lambda_iso");

    double lambda_aniso = 0.0;//10.0; //10.0;
    config.readInto(lambda_aniso, "lambda_aniso");

    double lambda_rot = 0.0;//10.0;
    config.readInto(lambda_rot, "lambda_rot");

    double kp    = 0.2*0.1;
    config.readInto(kp, "kp");

    double kd    = 0.1;
    config.readInto(kd, "kd");

    double fric    = 0.0;
    config.readInto(fric, "fric");

    double sus20    = 200.0;
    config.readInto(sus20, "sus20");

    double sus40    = 800.0;
    config.readInto(sus40, "sus40");

    double hcrit    = 0.25;
    config.readInto(hcrit, "hcrit");

    //Time integrator
    double deltat  = 1E-2;
    config.readInto(deltat, "deltat");

    //Numerical parameters for linear and NR solver
    string linTol = "1e-12";
    config.readInto(linTol, "linTol");

    double resTol = 1e-12;
    config.readInto(resTol, "resTol");

    double solTol = 1e-12;
    config.readInto(solTol, "solTol");

    int maxIter = 6;
    config.readInto(maxIter, "maxIter");

    string linImax = "10000";
    config.readInto(linImax, "linImax");

    double maxDelt = 1.0;
    config.readInto(maxDelt, "maxDelt");
    //testCase=1 for the annulus
    //testCase=2 for full circle
    int testCase = 1;
    config.readInto(testCase, "testCase");

    double kosm = 0.001;
    config.readInto(kosm, "kosm");

    double kgrad = 0.0;
    config.readInto(kgrad, "kgrad");

    int totalTimeSteps{1000000};
    config.readInto(totalTimeSteps, "totalTimeSteps");


    double adaptiveStepTime = 0.975;
    config.readInto(adaptiveStepTime, "adaptiveStepTime");

    int nPrint = 20;
    config.readInto(nPrint, "nPrint");

    double stab = 1.0;
    config.readInto(stab, "stab");

    double uPoly = -5.0/60.0;
    config.readInto(uPoly, "uPoly");

    double t_stall = -1.0;
    config.readInto(t_stall, "t_stall");

    double heqb = 0.20;
    config.readInto(heqb, "heqb");

    double L=0.0;
    config.readInto(L, "L");

    double theta= M_PI/2;
    config.readInto(theta, "theta");

    double lambda_trans = 6.24;
    config.readInto(lambda_trans, "lambda_trans");

    double kon_phi = 50;
    config.readInto(kon_phi, "kon_phi");

    double koff_phi = 5;
    config.readInto(koff_phi, "koff_phi");   

    double offset  =  6.5;
    
    double x_0 = offset; 
    config.readInto(x_0, "x_0");   

    double x_1 = 1 + offset; 
    config.readInto(x_1, "x_1");   

    double x_m = 0.5*(x_0 +x_1); 
    config.readInto(x_m, "x_m");

    double chi      = -11.0;
    config.readInto(chi, "chi");   

    double epsilon2 = 1E-2;
    config.readInto(epsilon2, "epsilon2");   

    double mobility = 0.067;
    config.readInto(mobility, "mobility");

    double D_u = 0.1;
    config.readInto(D_u, "D_u");

    //Consistency check
    bool cCheck{true};
    config.readInto(cCheck, "cCheck");

    ofstream fileNameDiss(csvPathDiss);
    fileNameDiss.close();

    //ofstream fileNameEnergy(csvPathEnergy);
    //fileNameEnergy.close();



    //Input
    string fMesh{};
    config.readInto(fMesh, "fMesh");

    //Output
    string solname_v;
    string oname = "stressFiberCircle";
    config.readInto(oname, "oname");
    int timeStep=restart;
    double time = 0.0;
    //Parameters of the mesh and geometry
    int bfOrder = 1;
    bool balanceMesh = true ;

    //Add model parameters to userStr

    SmartPtr<ParamStructure>  userNum = Create<ParamStructure>() ;
    SmartPtr<ParamStructure> userStr = Create<ParamStructure>() ;
    {
        userStr->dparam.resize(50);

        userStr->dparam[0] = deltat;
        userStr->dparam[1] = sus20;
        userStr->dparam[2] = sus40;
        userStr->dparam[3] = hcrit;

        userStr->dparam[4] = kp;
        userStr->dparam[5] = kd;
        userStr->dparam[6] = frank;

        userStr->dparam[7] = lambda_iso;
        userStr->dparam[8] = lambda_aniso;
        userStr->dparam[9] = lambda_rot;

        userStr->dparam[10] = visc;
        userStr->dparam[11] = rvisc;
        userStr->dparam[12] = cvisc;

        userStr->dparam[13] = fric;
        userStr->dparam[14] = stab;
        userStr->dparam[15] = phi;
        userStr->dparam[16] = r_in;
        userStr->dparam[17] = r_out;
        userStr->dparam[18] = uPoly;
        userStr->dparam[19] = t_stall;
        userStr->dparam[20] = kosm;
        userStr->dparam[21] = kgrad;        
        userStr->dparam[22] = heqb;
        userStr->dparam[23] = L;
        userStr->dparam[24] = lambda_trans;
        userStr->dparam[25] = theta;
        userStr->dparam[26]  = kon_phi;
        userStr->dparam[27]  = koff_phi;
        userStr->dparam[28]  = x_m;
        userStr->dparam[29]  = x_0;
        userStr->dparam[30]  = x_1;
        userStr->dparam[31]  = chi;
        userStr->dparam[32]  = epsilon2;
        userStr->dparam[33]  = mobility;
        userStr->dparam[34]  = D_u;
 
        //Numerical parameters
        userNum->dparam.resize(3);
        userNum->iparam.resize(3);
        userNum->dparam[0] = solTol;
        userNum->dparam[1] = resTol;
        userNum->iparam[0] = maxIter;
        userNum->iparam[1] = testCase;
    }
    RCP<MeshLoader> loadedMesh = rcp(new MeshLoader);
    if (testCase==1)
    {
        loadedMesh->setElemType(ElemType::Triang);
        loadedMesh->setBasisFuncType(BasisFuncType::SubdivSurfs);
        loadedMesh->setBasisFuncOrder(2);
        loadedMesh->loadMesh(fMesh, hiperlife::MeshType::Parallel);
    }

    //Dismesh
    RCP<DistributedMesh> disMesh = rcp(new DistributedMesh);
        if (testCase==1)
            disMesh->setMesh(loadedMesh);

        disMesh->setBalanceMesh(balanceMesh);
        disMesh->Update();
        disMesh->printFileLegacyVtk("DensityDependCortexNemacMesh");
        if (disMesh->myRank() == 0)
            cout << "Dismesh successfully created. " << endl;

    
    //DOFsHand
    SmartPtr<DOFsHandler>  dhand = Create<DOFsHandler>(disMesh);
    SmartPtr<DOFsHandler>  dofHand = Create<DOFsHandler>(disMesh);   

       try
       {
         dofHand->setNameTag("cortexHand");
         dofHand->setDOFs({"h", "q1", "q2", "vx", "vy","ux","uy"});
         dofHand->setNodeAuxF({"phi"});
         dofHand->Update();

         if (dofHand->myRank() == 0)
            cout << "DOFsHandler_velocity successfully created. " << endl;
       }
       catch (runtime_error err)
       {
         cout << dofHand->myRank() << ":   DOFsHandler_velocity could not be created. " << err.what() << endl;
         MPI_Finalize();
         return 1;
       }
       try
       {
            dhand->setNameTag("dhand");
            dhand->setDOFs({"phi"});
            dhand->setNodeAuxF({"vx","vy","ux","uy"});
            dhand->Update();
            cout << dhand->myRank() << ":   DOFsHandler_phi successfully created. " << endl;
        }
        catch (runtime_error& err)
        {
            cout << dhand->myRank() << ":   DOFsHandler_phi could not be created. " << err.what() << endl;

            MPI_Finalize();
            return 1;
        } 

       setSectorVelBC(dofHand, userStr);
       setSectorThiBC(dofHand, userStr);
       setSectorNemBC(dofHand, userStr);
       setSectorPhiBC(dhand,userStr);


       dofHand->nodeDOFs->setValue("ux",0.0);
       dofHand->nodeDOFs->setValue("uy",0.0);
       // ux/uy solved by FEM — no constraint
       dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
       dofHand->UpdateGhosts();
       dhand->nodeDOFs0->setValue(dhand->nodeDOFs);
       dhand->UpdateGhosts();


    //Hiperproblem
    RCP<HiPerProblem> hiperProbl;

    hiperProbl = rcp(new HiPerProblem);
    hiperProbl->setParameterStructure(userStr);
    hiperProbl->setDOFsHandlers({dofHand});
    hiperProbl->setConsistencyCheckTolerance(1.E-5);
    hiperProbl->setConsistencyCheckDelta(1.E-5);
    //Set integration
    hiperProbl->setIntegration("Integ", {"cortexHand"});
    hiperProbl->setIntegration("BorderInteg", {"cortexHand"});
    // annulus with a structured square mesh
    if (testCase==1)
    {
        hiperProbl->setCubatureGauss("Integ",  3);
        hiperProbl->setCubatureBorderGauss("BorderInteg", 2);
    }
    else if (testCase==2)
    {
        hiperProbl->setCubatureGauss("Integ", 3);
    }
    //Element fillings
    if (false)
    {
        hiperProbl->setElementFillings("Integ", ConsistencyCheck<LS_CortexNematic_2D>);
        hiperProbl->setElementFillings("BorderInteg", ConsistencyCheck<LS_CortexNematic_Border>);
    }
    else
    {
        hiperProbl->setElementFillings("Integ", LS_CortexNematic_2D);
        hiperProbl->setElementFillings("BorderInteg", LS_CortexNematic_Border);
    }
    //Update
    hiperProbl->Update();

    if (hiperProbl->myRank() == 0)
        cout << "HiperProblem successfully updated." << endl;


    // ElemType eType = ElemType::Square;
    // gPts= pow(2+1, computePDim(eType));
   // Create HiPerProblem
    SmartPtr<HiPerProblem> hiperProbl_phi = Create<HiPerProblem>();
    try
    {

        // Set DOFHandler
        hiperProbl_phi->setDOFsHandlers({dhand});
        // Set Integration
        hiperProbl_phi->setIntegration("Integ", {"dhand"});
        hiperProbl_phi->setCubatureGauss("Integ", 3);
        hiperProbl_phi->setElementFillings("Integ", LS_phi);
        hiperProbl_phi->setParameterStructure(userStr);
        hiperProbl_phi->setIntegration("BorderInteg", {"dofHand"});
        hiperProbl_phi->setCubatureBorderGauss("BorderInteg",2);
        hiperProbl_phi->setElementFillings("BorderInteg", LS_border_phi);
        // Set global integral
        hiperProbl_phi->setConsistencyCheckDelta(1.E-4);
        hiperProbl_phi->setConsistencyCheckTolerance(1.E-4);
        hiperProbl_phi->setConsistencyCheckType(ConsistencyCheckType::Hessian);
        // Update
        hiperProbl_phi->Update();
    }
    catch (runtime_error& err)
    {
        cout << hiperProbl_phi->myRank()  << ": HiPerProblem could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1;
    }


    //Output initial condition
    string solName = oname + ".0";
    if (printVtk)
        dofHand->printFileLegacyVtk(solName, true);
    if (printFile)
        dofHand->printFile(solName, OutputMode::Text, true, 0.0);
    // Open the file to store the global integrals
    fileNameDiss.open(csvPathDiss, ios::app);
    //fileNameEnergy.open(csvPathEnergy, ios::app);



    RCP<MUMPSDirectLinearSolver> linsolver =  rcp(new MUMPSDirectLinearSolver());
    linsolver->setHiPerProblem(hiperProbl);
    linsolver->setMatrixType(MUMPSDirectLinearSolver::MatrixType::General); //General, POD or Symmetric
    linsolver->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel); //Sequential or parallel
    linsolver->setOrderingLibrary(MUMPSDirectLinearSolver::OrderingLibrary::Auto); //Many options here
    linsolver->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linsolver->setDefaultParameters();
    linsolver->setWorkSpaceMemoryIncrease(1000);
    linsolver->Update();


    //Set non-lienar solver
    RCP<NewtonRaphsonNonlinearSolver> nonlinSolver =  rcp ( new NewtonRaphsonNonlinearSolver());
    nonlinSolver->setLinearSolver(linsolver);
    nonlinSolver->setMaxNumIterations(maxIter);
    nonlinSolver->setResTolerance(resTol);
    nonlinSolver->setSolTolerance(solTol);
    nonlinSolver->setLineSearch(false);
    nonlinSolver->setPrintIntermInfo(true);
    nonlinSolver->setConvRelTolerance(true);
    nonlinSolver->Update();

    // // Create linear solver
    RCP<MUMPSDirectLinearSolver> linSolver_phi =  rcp(new MUMPSDirectLinearSolver());
    linSolver_phi->setHiPerProblem(hiperProbl_phi);
    linSolver_phi->setMatrixType(MUMPSDirectLinearSolver::MatrixType::General); //General, POD or Symmetric
    linSolver_phi->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel); //Sequential or parallel
    linSolver_phi->setOrderingLibrary(MUMPSDirectLinearSolver::OrderingLibrary::Auto); //Many options here
    linSolver_phi->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linSolver_phi->setDefaultParameters();
    linSolver_phi->setWorkSpaceMemoryIncrease(1000);
    linSolver_phi->Update();


    // Create nonlinear solver
    SmartPtr<NewtonRaphsonNonlinearSolver> nonlinSolver_phi = Create<NewtonRaphsonNonlinearSolver>();
    nonlinSolver_phi->setLinearSolver(linSolver_phi);
    nonlinSolver_phi->setMaxNumIterations(6);
    nonlinSolver_phi->setResTolerance(1.E-6);
    nonlinSolver_phi->setSolTolerance(1.E-6);
    nonlinSolver_phi->setLineSearch(false);
    nonlinSolver_phi->setConvRelTolerance(false);
    nonlinSolver_phi->setPrintIntermInfo(true);
    nonlinSolver_phi->setPrintSummary(false);
    nonlinSolver_phi->Update();

    dofHand->printFileVtk("initial", true,0);
    dhand->printFileVtk("initial_phi", true,0);
    //Loop over time steps
    while (timeStep < totalTimeSteps)
    {
        //Write
        if (hiperProbl->myRank() == 0)
            cout << "    Time step " + to_string(timeStep) + " "
                 << "deltat= " << userStr->dparam[0] << " and maximum iteration " << maxIter << endl;

        //Initial guess
        dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
        dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
        hiperProbl->UpdateGhosts();
        hiperProbl_phi->UpdateGhosts();
        //Newton-Raphson method
        bool converged = nonlinSolver->solve();

        //Check convergence
        if (converged)
        {

            // Sync new v/u to dhand->nodeAuxF before phi solve
            for (int i = 0; i < dhand->mesh->loc_nPts(); ++i)
            {
                double ux = dofHand->nodeDOFs->getValue("ux", i, IndexType::Local);
                double uy = dofHand->nodeDOFs->getValue("uy", i, IndexType::Local);
                double vx = dofHand->nodeDOFs->getValue("vx", i, IndexType::Local);
                double vy = dofHand->nodeDOFs->getValue("vy", i, IndexType::Local);
                dhand->nodeAuxF->setValue("ux", i, IndexType::Local, ux);
                dhand->nodeAuxF->setValue("uy", i, IndexType::Local, uy);
                dhand->nodeAuxF->setValue("vx", i, IndexType::Local, vx);
                dhand->nodeAuxF->setValue("vy", i, IndexType::Local, vy);
            }
            dhand->UpdateGhosts();

            bool converged_phi = nonlinSolver_phi->solve();
            if (converged_phi)
            {

                //Save solution
                dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
                dhand->nodeDOFs0->setValue(dhand->nodeDOFs);

                for (int i = 0; i < dhand->mesh->loc_nPts(); ++i)
                {
                    double phi = dhand->nodeDOFs->getValue("phi", i, IndexType::Local);
                    dofHand->nodeAuxF->setValue("phi", i, IndexType::Local, phi);
                }
                dofHand->UpdateGhosts();
                dhand->UpdateGhosts();

                deltat /= adaptiveStepTime;
                //Update time variables
                timeStep++;
                time = time + deltat;

                //Modify time-step size
                if (deltat > maxDelt)
                    deltat = maxDelt;


                //Write results and update solution
                if (timeStep % nPrint == 0)
                {
                    solname_v = oname + "." + to_string(timeStep);
                    if (printVtk)
                    {
                        dofHand->printFileVtk(solname_v, true,time);    
                    }
                    // if (printFile)
                    // {
                    //     dofHand->printFile(solname_v, OutputMode::Text, true, time);
                    // }
                }
            }
            else
            {
                // phi failed — reset both fields to last converged state
                dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
                dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
                dofHand->UpdateGhosts();
                dhand->UpdateGhosts();
                deltat *= adaptiveStepTime;
            }

        }
        else
        {
            // v/u failed — reset both fields to last converged state
            dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
            dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
            dofHand->UpdateGhosts();
            dhand->UpdateGhosts();
            deltat *= adaptiveStepTime;
        }

        //Update deltat
        userStr->dparam[0] = deltat;

    }

    //Print last time step
    solname_v = oname + "." + to_string(timeStep+1);
    if (printVtk)
    {
        dofHand->printFileVtk(solname_v, true,time);
        //dhand->printFileVtk("phi."+ to_string(timeStep+1), true,time);        
    }
    if (printFile)
        dofHand->printFile(solname_v, OutputMode::Text, true, time+deltat);

    //Finalize
    MPI_Finalize();
    return 0;

}


