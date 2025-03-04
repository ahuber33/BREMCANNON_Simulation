/// BREMCANNONSimSteppingAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#include "BREMCANNONSimSteppingAction.hh"

using namespace CLHEP;

const G4String BREMCANNONSimSteppingAction::path = "../simulation_input_files/";

BREMCANNONSimSteppingAction::BREMCANNONSimSteppingAction(BREMCANNONSimGeometry *OptGeom, BREMCANNONSimRunAction *runaction, BREMCANNONSimEventAction *eventaction)
    : detector(OptGeom), runaction(runaction), eventaction(eventaction)
{
}

BREMCANNONSimSteppingAction::~BREMCANNONSimSteppingAction() {}
void BREMCANNONSimSteppingAction::UserSteppingAction(const G4Step *aStep)
{
  // ###################################
  //  Déclaration of functions/variables
  // ###################################
  Parent_ID = aStep->GetTrack()->GetParentID();
  StepNo = aStep->GetTrack()->GetCurrentStepNumber();
  track = aStep->GetTrack();
  thePrePoint = aStep->GetPreStepPoint();
  thePostPoint = aStep->GetPostStepPoint();
  ProcessName = thePostPoint->GetProcessDefinedStep()->GetProcessName();
  VolumName = track->GetVolume()->GetName();
  PreVolumName = thePrePoint->GetPhysicalVolume()->GetName();
  PartName = track->GetDefinition()->GetParticleName();
  ID = track->GetTrackID();
  prePoint = aStep->GetPreStepPoint()->GetPosition();
  postPoint = aStep->GetPostStepPoint()->GetPosition();
  point = prePoint + G4UniformRand() * (postPoint - prePoint);
  x = aStep->GetTrack()->GetPosition().x();
  y = aStep->GetTrack()->GetPosition().y();
  z = aStep->GetTrack()->GetPosition().z();
  Edep = aStep->GetTotalEnergyDeposit() / keV;
  Energy = track->GetKineticEnergy() / MeV;
  Labs = detector->GetLabs();
  Number_IP = detector->GetNumberIP();

  if (PartName == "geantino" && PreVolumName == "OpenDisk_Entry")
  {
    eventaction->ActiveCompteur();
    // G4cout << "Compteur = " << eventaction->GetCompteur() << G4endl;
  }


  if (Parent_ID == 0 && StepNo == 1)
  {
    eventaction->SetIncidentE(aStep->GetPreStepPoint()->GetKineticEnergy() / keV);
    runaction->FillIncidentHisto(aStep->GetPreStepPoint()->GetKineticEnergy() / keV);
  }



  if (StepNo == 1)
  {
    // G4cout << "ID = " << ID << G4endl;
    eventaction->SetIDPartName(PartName);
    if (Parent_ID == 0 && PartName == "gamma")
      eventaction->SetTrackingID(ID, PartName);

    if (Parent_ID > 0)
    {
      // G4cout << "Creator Process = " << track->GetCreatorProcess()->GetProcessName() << G4endl;
      if (track->GetCreatorProcess()->GetProcessName() == "compt" || track->GetCreatorProcess()->GetProcessName() == "phot" || track->GetCreatorProcess()->GetProcessName() == "conv" || track->GetCreatorProcess()->GetProcessName() == "eBrem")
      {
        eventaction->SetTrackingID(ID, "gamma");
      }
      else
      {
        for (int i = 0; i < eventaction->GetIDParentSize(); i++)
        {
          if (Parent_ID == eventaction->GetIDParent(i))
          {
            // G4cout << "ID[" << i << "] = " << eventaction->GetIDParent(i) << G4endl;
            eventaction->SetTrackingID(ID, "gamma");
            break;
          }
        }
      }
    }
  }
  

  std::string prefix1 = "Stack_IP1_";
  std::string prefix2 = "Stack_IP2_";

  if (PreVolumName.find(prefix1) == 0 || PreVolumName.find(prefix2) == 0)
  {
    for (G4int num_IP = 0; num_IP < Number_IP; ++num_IP)
    {
      std::string stack1 = "Stack_IP1_" + std::to_string(num_IP);
      std::string stack2 = "Stack_IP2_" + std::to_string(num_IP);

      if (PreVolumName == stack1)
      {
        if (PartName == "e-" && !eventaction->GetElectronFlag(num_IP))
        {
          eventaction->SetIncidentELecStack(num_IP, Energy);
          eventaction->ActiveElectronFlag(num_IP);
        }
        else if (PartName == "gamma")
        {
          // Handle gamma case if needed
        }
      }

      if (PreVolumName == stack2)
      {
        runaction->AddEnergyTot(num_IP, Edep);
        EdepEff = Edep * exp(-(point.z() - detector->GetPosZSensitive(num_IP)) / Labs);
        if (eventaction->GetIDPartName() == "e-")
        {
          runaction->AddElectronEnergy(num_IP, Edep);
          eventaction->ActiveFlagGoodEvent();
          runaction->AddEnergyTotEff(num_IP, EdepEff);
          runaction->AddElectronEnergyEff(num_IP, EdepEff);
        }
        else if (eventaction->GetIDPartName() == "gamma")
        {
          runaction->AddGammaEnergy(num_IP, Edep);
          eventaction->ActiveFlagGoodEvent();
          runaction->AddEnergyTotEff(num_IP, EdepEff);
          runaction->AddGammaEnergyEff(num_IP, EdepEff);
        }
        else
        {
          eventaction->ActiveFlagGoodEvent();
          runaction->AddEnergyTotEff(num_IP, EdepEff);
        }
      }
    }
  }

  if (Parent_ID > 0)
  {
    if (track->GetCreatorProcess()->GetProcessName() == "eBrem")
    {
      if (StepNo == 1)
      {
        // G4cout << "Brem with E = " << Energy << G4endl;
        eventaction->FillEBremCreation(Energy);
      }
      if (VolumName == "PhysicalWorld")
      {
        eventaction->FillEBremExit(Energy);
      }
    }
  }
}
