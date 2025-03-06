/// BREMCANNONSimMaterials.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimMaterials_h
#define BREMCANNONSimMaterials_h

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4Material.hh"
#include "G4NistManager.hh"

class G4Material;

class BREMCANNONSimMaterials
{
public:
  static BREMCANNONSimMaterials *getInstance();
  virtual ~BREMCANNONSimMaterials();
  G4Material *GetMaterial(const char*);

protected:
BREMCANNONSimMaterials();
  


private:
  std::vector<G4Material *> fMaterialsList;
  static BREMCANNONSimMaterials *fgInstance;

  G4String TypeIP;
  G4int Nbre_IP;

};
#endif
