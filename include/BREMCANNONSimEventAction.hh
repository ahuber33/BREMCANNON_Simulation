/// BREMCANNONSimEventAction.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimEventAction_h
#define BREMCANNONSimEventAction_h 1

#include "G4UserEventAction.hh"
#include "TH1F.h"
#include "TROOT.h"
#include "TTree.h"
#include "TBranch.h"

class G4Event;

// This struct carries statistics for the whole Run
struct RunTally
{
  float IncidentE;
  int compteur;
  std::vector<float> EBremCreation;
  std::vector<float> EBremExit;

  inline int operator==(const RunTally &right) const
  {
    return (this == &right);
  }
};

// Structure pour une pile unique
// struct StackTally {
//     float IncidentE_Elec;
//     float DepositE_Elec;
//     float EffectiveDepositE_Elec;
//     float X_Position_Elec;
//     float Y_Position_Elec;
//     std::vector<float> IncidentE_Gamma;
//     float DepositE_Gamma;
//     float EffectiveDepositE_Gamma;
//     float X_Position_Gamma;
//     float Y_Position_Gamma;
// };

// // Structure principale
// struct RunTally {
//     float IncidentE; // Énergie incidente globale
//     std::vector<StackTally> Stacks; // Contient les données pour chaque pile
//     int compteur;
//     std::vector<float> EBremCreation;
//     std::vector<float> EBremExit;

//     // Constructeur pour initialiser avec un nombre de piles
//     RunTally(size_t n) : Stacks(n) {}

//     // Comparaison pour l'égalité
//     inline bool operator==(const RunTally& right) const {
//         return (this == &right);
//     }

class BREMCANNONSimEventAction : public G4UserEventAction
{
public:
  BREMCANNONSimEventAction(const char *);
  ~BREMCANNONSimEventAction();

public:
  void BeginOfEventAction(const G4Event *);
  void EndOfEventAction(const G4Event *);
  void SetTrackingID(G4int Track_ID, G4String PartName);
  G4int GetIDParentSize() { return TrackID.size(); }
  G4int GetIDParent(G4int i) { return TrackID.at(i); }
  G4String GetIDPartName() { return ParticuleName; }
  void SetIDPartName(G4String PartName) { ParticuleName = PartName; }
  void ActiveElectronFlag(G4int num_IP) { electron_flag_incident[num_IP] = true; }
  G4bool GetElectronFlag(G4int num_IP) { return electron_flag_incident[num_IP]; }

  void SetIncidentE(G4double a) { Statistics.IncidentE = a; }
  void SetIncidentELecStack(G4int num_IP, G4double a) { IncidentE_Elec_Stack[num_IP] = a; }

  //   void AddEdepElec(size_t stackIndex, G4double a) {
  //     if (stackIndex < Statistics.Stacks.size()) {
  //         Statistics.Stacks[stackIndex].DepositE_Elec += a;
  //     } else {
  //         // Gérer le cas où l'index est hors limites
  //         std::cerr << "Erreur : index de pile " << stackIndex << " hors limites !" << std::endl;
  //     }
  // }

  void ActivePositionElectronFlag(G4int num_IP) { electron_flag_position[num_IP] = true; }
  G4bool GetPositionElectronFlag(G4int num_IP) { return electron_flag_position[num_IP]; }

  void ActiveFlagGoodEvent() { flag_good_event = true; }
  void ActivePositionGammaFlag(G4int num_IP) { gamma_flag_position[num_IP] = true; }
  G4bool GetPositionGammaFlag(G4int num_IP) { return gamma_flag_position[num_IP]; }
  void ActiveCompteur() { Statistics.compteur++; }
  G4int GetCompteur() { return Statistics.compteur; }
  void FillEBremCreation(G4double a) { Statistics.EBremCreation.push_back(a); }
  void FillEBremExit(G4double a) { Statistics.EBremExit.push_back(a); }

private:
  TTree *EventTree;
  TBranch *EventBranch;
  RunTally Statistics;
  G4String suffixe;
  std::vector<int> TrackID;
  G4String ParticuleName;
  G4bool electron_flag_incident[20];
  G4bool electron_flag_position[20];
  G4bool gamma_flag_position[20];
  G4bool flag_good_event;
  G4double IncidentE_Elec_Stack[20];
  G4double DepositE_Elec_Stack[20];
  G4double DepositE_Gamma_Stack[20];
};

#endif
