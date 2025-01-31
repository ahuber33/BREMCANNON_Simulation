/// BREMCANNONSimActionInitialization.hh
//// Auteur: Arnaud HUBER for ENL group <huber@lp2ib.in2p3.fr>
//// Copyright: 2024 (C) Projet PALLAS

#ifndef BREMCANNONSimActionInitialization_h
#define BREMCANNONSimActionInitialization_h 1

#include "G4VUserActionInitialization.hh"
#include "globals.hh"
#include "BREMCANNONSimPrimaryGeneratorAction.hh"
#include "BREMCANNONSimRunAction.hh"
#include "BREMCANNONSimEventAction.hh"
#include "BREMCANNONSimSteppingAction.hh"
#include "BREMCANNONSimGeometry.hh"
#include "G4MTRunManager.hh"


class BREMCANNONSimGeometryConstruction;
class BREMCANNONSimPrimaryGeneratorAction;

class BREMCANNONSimActionInitialization : public G4VUserActionInitialization
{
public:
  BREMCANNONSimActionInitialization(BREMCANNONSimGeometry*, const char*, G4bool pMT);
  virtual ~BREMCANNONSimActionInitialization();
  size_t charToSizeT(G4String str);

  virtual void BuildForMaster() const;
  virtual void Build() const;
  char* NEvents;
  G4String suffixe;
  G4bool flag_MT=false;

private:
BREMCANNONSimGeometry * detector;
  
};

#endif