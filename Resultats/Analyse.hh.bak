const int n=21;
float Nsim = 1e7;
char file[100];
TH1D* h;

float IP1[n];
float IP2[n];
float IP3[n];
float IP4[n];
float IP5[n];
float IP6[n];
float IP7[n];
float IP8[n];
float IP9[n];
float IP10[n];
float IP11[n];
float IP12[n];
float IP13[n];
float IP14[n];
float IP15[n];



TH1D* GetGraph(const char* filename)
{
  TFile* file = new TFile(filename);
  TH1D* f = (TH1D*)file->Get("edep_totale");

  return f;
}

float GetBinValue(int bin)
{
  float a = h->GetBinContent(bin);
  if(a<3) a=2.3;
  float value = a/Nsim;

  return value;
  
}
