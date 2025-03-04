/// BREMCANNONSimRunAction.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimRunAction_h
#define BREMCANNONSimRunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"
#include "BREMCANNONSimEventAction.hh"
#include "BREMCANNONSimGeometry.hh"
#include "Randomize.hh"
#include "G4Run.hh"
#include "G4UImanager.hh"
#include "G4VVisManager.hh"
#include "TFile.h"

//class Geometry;
class BREMCANNONSimGeometry;
class G4Run;

class BREMCANNONSimRunAction : public G4UserRunAction
{

public:
  BREMCANNONSimRunAction(BREMCANNONSimGeometry*, const char*, G4bool);
  ~BREMCANNONSimRunAction();

public:
  void DrawMemoryInformations();
  void CreateROOTFile(G4String);
  void WriteROOTFile(TFile* f);
  void MergeHistograms();
  void DisplayRunTime(time_t, time_t);
  void BeginOfRunAction(const G4Run*);
  void EndOfRunAction(const G4Run*);
  void EcrireDistribElectrons(G4int num_IP, G4double Energie);
  void EcrireDistribGammas(G4int num_IP, G4double Energie);
  void SetHistoName(G4String name) {fname = name;}
  void SetStepParams(G4int m_ID,G4double m_Temps,G4String m_PartName);
  void InitialiseHistos();
  void FillIncidentHisto(G4double e) {hist->Fill(e);}

  void AddElectronEnergy(G4int num_IP, G4double Edep) {E_dep_electrons[num_IP]+=Edep;}
  void AddGammaEnergy(G4int num_IP, G4double Edep) {E_dep_gammas[num_IP]+=Edep;}
  void AddEnergyTot(G4int num_IP, G4double Edep) {E_dep_tot[num_IP]+=Edep;}
  void AddElectronEnergyEff(G4int num_IP, G4double Edep) {E_dep_electrons_eff[num_IP]+=Edep;}
  void AddGammaEnergyEff(G4int num_IP, G4double Edep) {E_dep_gammas_eff[num_IP]+=Edep;}
  void AddEnergyTotEff(G4int num_IP, G4double Edep) {E_dep_tot_eff[num_IP]+=Edep;}

  //adds the photon fates from an event to the run tree
  void UpdateStatistics(RunTally);


private:
  BREMCANNONSimGeometry * detector;
  static const G4int Nombre_IP = 15;
  static const G4String path_bin;
  int threadID;
  static G4Mutex fileMutex;
  G4String suffixe;
  G4String fileName;
  G4String histoFileName;
  RunTally Stats;
  G4bool flag_MT;
  G4long seed;
  TFile *f;
  TFile *histoFile;
  TTree *IP;
  TBranch *RunBranch;
  time_t start;
  time_t end;
  TH1D* hist;
  G4String fname, nom_fichier;
  TH1D *HistoDistribElectrons[Nombre_IP];
  TH1D *HistoDistribGammas[Nombre_IP];
  G4double E_dep_electrons[Nombre_IP],  E_dep_gammas[Nombre_IP], E_dep_tot[Nombre_IP];
  G4double E_dep_electrons_eff[Nombre_IP],  E_dep_gammas_eff[Nombre_IP], E_dep_tot_eff[Nombre_IP];
  G4double Rayon_cylindre_interieur;
  G4double Rayon_cylindre_exterieur;
  G4double Stack_IP_pos_ini;
  G4double Disque_epaisseur;
  G4double Bloc_mylar_hauteur;
  G4double Cylindre_tantale_hauteur;
  G4double IPa1_z;
  G4double IPa2_z;
  G4double IPa3_z;
  G4double IPa4_z;
  G4double PosZ_Sensitive[Nombre_IP];
  G4double Epaisseur_Filtre[Nombre_IP];
  G4double Labs;
  G4double Coef_PSL;

};

#endif
