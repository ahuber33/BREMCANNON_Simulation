/// BREMCANNONSimMaterials.cc
/// Auteur: Arnaud HUBER <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#include "BREMCANNONSimMaterials.hh"

using namespace CLHEP;

BREMCANNONSimMaterials *BREMCANNONSimMaterials::fgInstance = nullptr;


BREMCANNONSimMaterials::BREMCANNONSimMaterials() : fMaterialsList{}
{
	G4double a, z, fractionmass;
	G4double density, pressure, temperature;
	G4double R = 8.3144621; //(en J/K/mole => Constante gaz parfait)
	G4int nel;
	density = 1.e-25 * g / cm3;
	pressure = 1.e-19 * pascal;
	temperature = 0.1 * kelvin;

	// Hydrogen
	auto elementH = new G4Element("Hydrogen", "H", z = 1., a = 1.0079 * g / mole);
	// Carbone
	auto elementC = new G4Element("Carbon", "C", z = 6., a = 12.01 * g / mole);
	// Oxygen
	auto elementO = new G4Element("Oxygen", "O", z = 8., a = 15.9994 * g / mole);
	// Baryum
	auto elementBa = new G4Element("Barium", "Ba", z = 56., a = 138.9055 * g / mole);
	// Fluor
	auto elementF = new G4Element("Fluorine", "F", z = 9., a = 18.9984 * g / mole);
	// Brome
	auto elementBr = new G4Element("Boron", "Br", z = 35., a = 79.904 * g / mole);
	// Azote
	auto elementN = new G4Element("Nitrogen", "N", z = 7., a = 14.0067 * g / mole);
	// Fe
	auto elementFe = new G4Element("Iron", "Fe", z = 26., a = 55.847 * g / mole);
	// Manganese
	auto elementMn = new G4Element("Manganese", "Mn", z = 25., a = 54.9380 * g / mole);
	// Zinc
	auto elementZn = new G4Element("Zinc", "Zn", z = 30., a = 65.38 * g / mole);

	//***********************
	// Build Materials      *
	//***********************
	TypeIP = "MS";

	if (TypeIP == "SR")
		density = 1.273 * g / cm3;
	else
		density = 1.66 * g / cm3;

	auto Polyester = new G4Material("Polyester", density = 1.66 * g / cm3, nel = 3);
	Polyester->AddElement(elementC, fractionmass = 0.625);
	Polyester->AddElement(elementH, fractionmass = 0.042);
	Polyester->AddElement(elementO, fractionmass = 0.333);
	fMaterialsList.push_back(Polyester);

	auto Sensitive = new G4Material("Sensitive", density = 3.31 * g / cm3, nel = 7);
	Sensitive->AddElement(elementC, fractionmass = 0.127);
	Sensitive->AddElement(elementH, fractionmass = 0.016);
	Sensitive->AddElement(elementO, fractionmass = 0.042);
	Sensitive->AddElement(elementN, fractionmass = 0.018);
	Sensitive->AddElement(elementBa, fractionmass = 0.463);
	Sensitive->AddElement(elementF, fractionmass = 0.064);
	Sensitive->AddElement(elementBr, fractionmass = 0.27);
	fMaterialsList.push_back(Sensitive);

	auto Base = new G4Material("Base", density = 2.77 * g / cm3, nel = 7);
	Base->AddElement(elementC, fractionmass = 0.121);
	Base->AddElement(elementH, fractionmass = 0.015);
	Base->AddElement(elementO, fractionmass = 0.354);
	Base->AddElement(elementN, fractionmass = 0.018);
	Base->AddElement(elementFe, fractionmass = 0.308);
	Base->AddElement(elementMn, fractionmass = 0.119);
	Base->AddElement(elementZn, fractionmass = 0.065);
	fMaterialsList.push_back(Base);
}

BREMCANNONSimMaterials::~BREMCANNONSimMaterials()
{
}

G4Material *BREMCANNONSimMaterials::GetMaterial(const char *materialId)
{
	for (int i = 0; i < (int)fMaterialsList.size(); i++)
	{
		if (fMaterialsList[i]->GetName() == materialId)
		{
			G4cout << "Material : " << materialId << " found" << G4endl;
			return fMaterialsList[i];
		}
	}
	G4cout << "ERROR: Materials::getMaterial material " << materialId << " not found." << G4endl;
	return NULL;
}

BREMCANNONSimMaterials *BREMCANNONSimMaterials::getInstance()
{
	static BREMCANNONSimMaterials materials;
	if (fgInstance == nullptr)
	{
		fgInstance = &materials;
	}
	return fgInstance;
}
