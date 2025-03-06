// Geometry.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#include "Geometry.hh"


using namespace CLHEP;

// ***********************
// Constructor
// ***********************
const G4String Geometry::path_bin = "../bin/";

// Fonction pour lire un tableau entre accolades {valeurs}
template <typename T>
std::vector<T> ParseArray(const std::string &line)
{
  std::vector<T> values;
  std::string content = line.substr(line.find("{") + 1, line.find("}") - line.find("{") - 1); // Récupère les valeurs entre {}
  std::stringstream ss(content);
  T value;

  while (ss >> value)
  {
    values.push_back(value); // Ajoute les valeurs au vecteur
    if (ss.peek() == ',')    // Ignore les virgules
      ss.ignore();
  }

  return values;
}

Geometry::Geometry(G4String buildfile)
{
  // Read keys and values from file buildfile defined in x_blockGeometry.cc
  // Some of these variables may not be needed.
  G4cout << "\n The Variables that we read in are: " << G4endl;

  std::ifstream config;
  config.open(buildfile);
  if (!config.is_open())
    G4cout << "Error opening file " << buildfile << G4endl;
  else
  {
    while (config.is_open())
    {
      G4String variable;
      G4String unit;
      G4double value;

      config >> variable;
      if (!config.good())
        break;
      // ####################### COMMON variables ###########################
      if (variable == "Radius_cylinder_internal")
      {
        config >> value >> unit;
        Radius_cylinder_internal = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Radius_cylinder_external")
      {
        config >> value >> unit;
        Radius_cylinder_external = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Radius_IP")
      {
        config >> value >> unit;
        Radius_IP = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Cylinder_tantale_length")
      {
        config >> value >> unit;
        Cylinder_tantale_length = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Disk_thickness")
      {
        config >> value >> unit;
        Disk_thickness = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Mylar_thickness")
      {
        config >> value >> unit;
        Mylar_thickness = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "flag_Mylar")
      {
        config >> value;
        flag_Mylar = value;
      }
      else if (variable == "Number_IP")
      {
        config >> value;
        Number_IP = value;
      }
      else if (variable == "IPa1_z")
      {
        config >> value >> unit;
        IPa1_z = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "IPa2_z")
      {
        config >> value >> unit;
        IPa2_z = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "IPa3_z")
      {
        config >> value >> unit;
        IPa3_z = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "IPa4_z")
      {
        config >> value >> unit;
        IPa4_z = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Thickness_filter")
      {
        std::string line;
        std::getline(config, line);             // Récupère toute la ligne
        auto values = ParseArray<double>(line); // Remplit le tableau
        if (values.size() != Number_IP - 1)
        {
          G4Exception("Config", "config0001", FatalException,
                      "Number of filters data != (Number IP-1)");
        }
        for (size_t i = 0; i < values.size(); ++i)
        {
          Thickness_Filter[i] = values[i] * G4UnitDefinition::GetValueOf("mm");
          G4cout << "Thickness_Filter [" << i << "] = " << Thickness_Filter[i] << G4endl;
        }
      }
      else if (variable == "Material_filter")
      {
        std::string line;
        std::getline(config, line);
        auto materials = ParseArray<std::string>(line);
        if (materials.size() != Number_IP - 1)
        {
          G4Exception("Config", "config0002", FatalException,
                      "Number of filters materials != (Number IP-1)");
        }
        for (size_t i = 0; i < materials.size(); ++i)
        {
          Material_Filter[i] = materials[i]; // Stocke les matériaux
          G4cout << "Material_Filter [" << i << "] = " << Material_Filter[i] << G4endl;
        }
      }
      else if (variable == "Labs")
      {
        config >> value >> unit;
        Labs = value * G4UnitDefinition::GetValueOf(unit);
      }
      else if (variable == "Coef_PSL")
      {
        config >> value >> unit;
        Coef_PSL = value;
      }
    }
  }
  config.close();

  G4cout << "\n Radius_cylinder_internal = " << Radius_cylinder_internal
         << "\n Radius_cylinder_external = " << Radius_cylinder_external
         << "\n Radius_IP = " << Radius_IP
         << "\n Disk_thickness = " << Disk_thickness
         << "\n Mylar_thickness = " << Mylar_thickness
         << "\n Flag Mylar = " << flag_Mylar
         << "\n Cylinder_tantale_length = " << Cylinder_tantale_length
         << "\n Number IP = " << Number_IP
         << "\n IPa1_z = " << IPa1_z
         << "\n IPa2_z = " << IPa2_z
         << "\n IPa3_z = " << IPa3_z
         << "\n IPa4_z = " << IPa4_z
         << "\n Labs = " << Labs
         << "\n Coef PSL = " << Coef_PSL

         << "\n " << G4endl;
}
// ***********************
// Destructor
// ***********************
Geometry::~Geometry()
{}

G4LogicalVolume *Geometry::GetFilter(G4String name, G4double Radius, G4double Thickness, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius, (Thickness / 2) * mm, 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetStackIP1(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius_IP, IPa1_z / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetStackIP2(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius_IP, IPa2_z / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetStackIP3(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius_IP, IPa3_z / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetStackIP4(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius_IP, IPa4_z / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetCorpus(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, Radius_cylinder_internal, Radius_cylinder_external, Cylinder_tantale_length / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetFullDisk(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0., Radius_cylinder_external, Disk_thickness / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetOpenDisk(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, Radius_cylinder_internal, Radius_cylinder_external, Disk_thickness / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetEntryProtection(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0, Radius_cylinder_internal, (Mylar_thickness / 2.) * mm, 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetVolumeFiller(G4String name, G4Material *Material, G4double length)
{

  auto solid = new G4Tubs("solid_"+name, 0, Radius_IP, length / 2., 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}

G4LogicalVolume *Geometry::GetPlaqueTantale(G4String name, G4Material *Material)
{

  auto solid = new G4Tubs("solid_"+name, 0, 1 * mm, 2.3 / 2 * mm, 0., 360. * deg);
  auto LogicalVolume = new G4LogicalVolume(solid, Material, "logical_"+name);

  return LogicalVolume;
}
