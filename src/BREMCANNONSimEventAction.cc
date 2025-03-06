/// BREMCANNONSimEventAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
/// Copyright: 2017 (C) Projet BADGE - CARMELEC -P2R

#include "G4SteppingManager.hh"
#include "BREMCANNONSimSteppingAction.hh"
#include "G4Run.hh"
#include "BREMCANNONSimEventAction.hh"
#include "G4RunManager.hh"

using namespace CLHEP;


BREMCANNONSimEventAction::BREMCANNONSimEventAction(const char* suff):suffixe(suff){}

BREMCANNONSimEventAction::~BREMCANNONSimEventAction(){}


// Initialize all counters and set up the event branches for
// filling histograms with ROOT
void BREMCANNONSimEventAction::BeginOfEventAction(const G4Event* evt){

  G4int event_id = evt->GetEventID();

  //G4cout << "EVENT : " << event_id << G4endl;

  Statistics = {};
  
  flag_good_event=false;

  for(G4int i = 0; i<10;i++){
    electron_flag_incident[i]=false;
    electron_flag_position[i] = false;
    gamma_flag_position[i] = false;
    IncidentE_Elec_Stack[i] =0;
    DepositE_Elec_Stack[i] =0;
    DepositE_Gamma_Stack[i] =0;
  }

TrackID.clear();

}

void BREMCANNONSimEventAction::SetTrackingID(G4int Track_ID, G4String PartName){
  TrackID.push_back(Track_ID);
  ParticuleName = PartName;
}

// Get the number of stored trajectories and calculate the statistics
void BREMCANNONSimEventAction::EndOfEventAction(const G4Event* evt){
  G4int event_id = evt->GetEventID();

  BREMCANNONSimRunAction *runac = (BREMCANNONSimRunAction*)(G4RunManager::GetRunManager()->GetUserRunAction());

  float Energie=0;
  for(G4int i = 0; i<20;i++){
    Energie+=DepositE_Elec_Stack[i];
    Energie+=DepositE_Gamma_Stack[i];
  }

    //if(flag_good_event==true)
    runac->UpdateStatistics(Statistics);
}
