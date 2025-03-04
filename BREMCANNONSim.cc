/// BREMCANNONSim.cc
/// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
/// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA


#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4ThreeVector.hh"
#include "G4MaterialPropertiesTable.hh"
#include "Randomize.hh"
#include "time.h"
#include "G4Timer.hh"
#include "G4UIterminal.hh"
#include "BREMCANNONSimSteppingAction.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
//#include "G4UIGAG.hh"
  #include "BREMCANNONSimGeometry.hh"
  #include "BREMCANNONSimPhysics.hh"
  #include "BREMCANNONSimPrimaryGeneratorAction.hh"
#include "BREMCANNONSimRunAction.hh"
#include "BREMCANNONSimEventAction.hh"
#include "BREMCANNONSimActionInitialization.hh"

//
#include "G4VisExecutive.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4RegionStore.hh"
#include "G4GeometryManager.hh"
#include "G4UIExecutive.hh"
#include "G4RunManagerFactory.hh"
#include "G4TaskRunManager.hh"


void CleanUp(BREMCANNONSimGeometry *geom)
{  
  G4GeometryManager::GetInstance()->OpenGeometry();

  // Suppression manuelle si nécessaire
  G4PhysicalVolumeStore::Clean();
  G4LogicalVolumeStore::Clean();
  G4SolidStore::Clean();
  G4RegionStore::Clean();
}

int main(int argc,char** argv){

  G4RunManager *runManager;
  const char *outputFile = argv[1];
  size_t NEvents = 1;
  size_t Nthreads = std::thread::hardware_concurrency();
  bool flag_MT = false;

  if (argc == 2) // VISUALIZATION MODE
  {
    runManager = new G4RunManager;
  }

  else if (argc > 4 && argc <= 6) // BATCH MODE
  {
    NEvents = std::stoul(argv[2]);
    G4String pMT = argv[4];

    if (pMT == "ON")
    {
      flag_MT = true;
      //runManager = new G4MTRunManager;
      runManager = G4RunManagerFactory::CreateRunManager();
      // Cast au type G4TaskRunManager pour accéder aux méthodes spécifiques
    auto* taskRunManager = static_cast<G4TaskRunManager*>(runManager);

      if (argc == 6)
        Nthreads = std::stoul(argv[5]);
      runManager->SetNumberOfThreads(Nthreads);
      //taskRunManager->SetGrainsize(10000); // 50 tâches
    }

    else if (pMT == "OFF")
    {
      flag_MT = false;
      runManager = new G4RunManager;
    }

    else
    {
      G4Exception("Main", "main0001", FatalException,
                  "MT parameter (5th argument) must be ON or OFF.");
    }
  }

  else
  {
    G4Exception("Main", "main0002", FatalException,
                "Incorrect number of input parameters.");
  }

  // set mandatory initialization classes
  BREMCANNONSimGeometry* Geom = new BREMCANNONSimGeometry;

  G4cout<<"Geometry given to BREMCANNONSim.cc"<<G4endl;

  // initialize the geometry
  runManager->SetUserInitialization(Geom);
  G4cout<<"Geometry set in BREMCANNONSim.cc given to Runman"<<G4endl;

  // initialize the physics
  runManager->SetUserInitialization(new BREMCANNONSimPhysics);

    // Initialize user actions
  runManager->SetUserInitialization(new BREMCANNONSimActionInitialization(Geom, outputFile, flag_MT));


//#ifdef G4VIS_USE
      G4VisManager* visManager = new G4VisExecutive;
      visManager->Initialize();
//#endif

  // Initialize G4 kernel
  runManager->Initialize();


  G4cout<<"Initialized new Run Manager"<<G4endl;

  // get the pointer to the User Interface manager
  G4UImanager* UI = G4UImanager::GetUIpointer();

  G4UIExecutive *ui = 0;

   if (argc == 2)
  {
    G4cout << "Interactive MODE" << G4endl;

    // #ifdef G4UI_USE
    ui = new G4UIExecutive(argc, argv);
    UI->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
    // #endif
  }

  else if (argc > 4) // batch mode
  {
    G4cout << "Batch MODE" << G4endl;

    G4String command = "/control/execute ";
    G4String macroName = argv[3];
    UI->ApplyCommand(command + macroName);
    UI->ApplyCommand("control/suppressAbortion");

    std::string runCommand = "/run/beamOn " + std::string(argv[2]);
    UI->ApplyCommand(runCommand);

    if (flag_MT)
    {
      std::string mergeCommand = "/control/shell hadd -k -f " + std::string(outputFile) + ".root";
      for (size_t i = 0; i < Nthreads; ++i)
      {
        mergeCommand += " " + std::string(outputFile) + "_" + std::to_string(i) + ".root";
      }
      UI->ApplyCommand(mergeCommand);

      // Clean up temporary files
      for (size_t i = 0; i < Nthreads; ++i)
      {
        std::string removeCommand = "/control/shell rm -f " + std::string(outputFile) + "_" + std::to_string(i) + ".root";
        UI->ApplyCommand(removeCommand);
      }
    }
  }

  else // define visualization and UI terminal for interactive mode
  {
    G4Exception("Main", "main0003", FatalException,
                "Problem with launch command");
  }


#ifdef G4VIS_USE
      delete visManager;
#endif


  std::string movefile = "/control/shell mv " + std::string(outputFile) + ".root ../Resultats";
  UI->ApplyCommand(movefile);
  G4cout << "Output saved in Resultats folder to file " << outputFile << ".root" << G4endl;

  // Appel de la fonction de nettoyage avant de quitter le programme
  CleanUp(Geom);

  delete runManager;
  return 0;
}
