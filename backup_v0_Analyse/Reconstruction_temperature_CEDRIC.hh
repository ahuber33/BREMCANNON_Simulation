TH1D* PSL_REF;
TH1D* Edep_REF;
TH1D* h;
TFile* file;
TH2F* Matrice = new TH2F("matrice", "matrice", 15, 0.5, 15.5, 100, 0, 100);
TH1D* PSL_Rec = new TH1D ("PSL_Rec", "PSL_Rec", 15, 0.5, 15.5);
TH1D* h_e;
TH1D* h_g;
TH1D* h_g_2T;
TH1D* Data = new TH1D ("Data", "Data", 15, 0.5, 15.5);
TH1F *Fit = new TH1F("fit", "fit", 100, 0, 100);
TH1D* REF;
float Chi2=0;
double A=0;
double err_A=0;
double E0=0;
double err_E0=0;
double Acold=0;
double err_Acold=0;
double Ecold=0;
double err_Ecold=0;
double Ahot=0;
double err_Ahot=0;
double Ehot=0;
double err_Ehot=0;
double NEvents=0;
double Coef_PSL = 6.95E-1;
//double Coef_PSL = 1.3E-1;
int flag_electron = 0;
float QL[15];
float IP[15];
TPad *pad1;
TPad *pad2;
TCanvas* c;
TF1* f1;
float Chi2_gamma_store=0;
float Chi2_gamma_2T_store=0;
float Chi2_store=0;
double N=0;
double eN=0;
float solid_angle=0;
double Surface =0;

float start_norm = 1e5;
float step_norm = 1e4;
float low_norm = 1e1;
float up_norm = 1e9;

float start_E = 40;//10
float step_E = 0.1;
float low_E = 0.1; //-1
float up_E = 800; //30

float start_Ecold = 10;
float step_Ecold = 1;
float low_Ecold = 1;
float up_Ecold = 30; 

float start_Ehot = 50;
float step_Ehot = 1;
float low_Ehot = 20; 
float up_Ehot = 1000;

float Tension_PMT = 560; //V
float Resolution = 50; //en µm
float Sensibilite =0;
float Latitude = 5;
float Dynamique = 16; //bits


float Calcul_Sensibilite(float V)
{
  // Source : www.osti.gov/servlets/purl/1227002
  float V0 = 500; //V
  float A0 = 0.008; //PSL
  float A1 = 1.2369; //PSL
  float A2 = 46.54; //V
  float A3 = 1.967; //PSL
  float A4 = 128.0; //V
  
  Sensibilite = A0 + A1*exp(-(V-V0)/A2) + A3*exp(-(V-V0)/A4);

  //cout << "Sensibilite (@ " << V << " Volts) = " << Sensibilite << endl;

  return Sensibilite;
}


float QL_to_PSL(float QL)
{
  //QL = QL/0.0025;
  Sensibilite = Calcul_Sensibilite(Tension_PMT);
  //float PSL = ((Resolution/100)*(Resolution/100)) * (4000/Sensibilite) * pow(10, Latitude*(QL/(pow(2, Dynamique-1)) - 1/2)) ; //These Thomas
  float PSL = pow(QL/(pow(2, 16)-1), 2)*pow(Resolution/100, 2)*Sensibilite*pow(10, Latitude/2);

  return PSL;
}




void Test()
{
  QL[0] = 41.1e4;
  QL[1] = 12e4;
  QL[2] = 5360e1;
  QL[3] = 2190e1;
  QL[4] = 760e1;
  QL[5] = 402e1;
  QL[6] = 122e1;
  QL[7] = 350;
  QL[8] = 118;
  QL[9] = 49;
  QL[10] = 5.3;
  QL[11] = 0.3;
  QL[12] = 0.15;
  QL[13] = 0;
  QL[14] = 0;
	

  cout << "Test program with simulation values" << endl;
}


void Test1()
{
  QL[0] = 1391398;
  QL[1] = 889543;
  QL[2] = 981726;
  QL[3] = 965038;
  QL[4] = 927195;
  QL[5] = 941925;
  QL[6] = 814710;
  QL[7] = 820880;
  QL[8] = 822584;
  QL[9] = 507158;
  QL[10] = 758017;
  QL[11] = 811098;
  QL[12] = 532833;
  QL[13] = 306659;
  QL[14] = 136442;
	

  cout << "Test program with simulation values" << endl;
}


void Create_Pad_Canvas(const char* name)
{
  gStyle->SetOptTitle(kFALSE);
  c = new TCanvas(name, name, 0, 10, 1600, 700);
  c->SetWindowPosition(600, 0);
  pad1 = new TPad("pad1", "", 0.0, 0.0, 0.5, 1);
  pad2 = new TPad("pad2", "", 0.51, 0.0, 1, 1);

  pad1->Draw();
  pad2->Draw();

}


void Init_IP()
{
  for (int i=0; i<15; i++)
  {
    QL[i] =0;
    IP[i] =0;
  }
}


double NumberOfEntries(const char* filename)
{
  TFile* file = new TFile(filename);
  TTree* Tree = (TTree*)file->Get("IP");
  int Entries = Tree->GetEntries();

  file->Close();

  return Entries;
}


void DefineMatrice(double coef_PSL, double Surface)
{
  TFile* f_Matrice = new TFile("CEDRIC_Matrice_LULI.root");
  Matrice = (TH2F*)f_Matrice->Get("matrice");
  Matrice->Scale(Coef_PSL/Surface);
}


void DefineMatriceElectron(double coef_PSL, double Surface)
{
  TFile* f_Matrice = new TFile("CEDRIC_Matrice_LULI.root");
  Matrice = (TH2F*)f_Matrice->Get("matrice");
  Matrice->Scale(Coef_PSL/Surface);
}


TH1D* Histo_PSL(const char* filename)
{
  file = new TFile(filename, "update");
  //PSL_REF = (TH1D*)file->Get("PSL vs IP");
  Edep_REF = (TH1D*)file->Get("edep_totale_eff");
  //Edep_REF = (TH1D*)file->Get("edep_totale");
  //Edep_REF->Scale(0.001);
  //PSL_REF->Scale(1/(3.1415927*7*7));

  int n = Edep_REF->GetNbinsX();
  NEvents = NumberOfEntries(filename);
  double Integral = Edep_REF->Integral();
  double Error_Edep=0;
  double Error_PSL=0;
  double Coef_PSL = 6.95E-1;

   for (int i=1; i<=n; i++)
     {
       Error_Edep = sqrt((Edep_REF->Integral(i,i)*Integral)/NEvents); //si graph Edep
       //Error_PSL = (Error_Edep*Coef_PSL)/(3.1415927*7*7); // si graph PSL
       Edep_REF->SetBinError(i, Error_Edep);
       //PSL_REF->SetBinError(i, Error_PSL);
     }


   //PSL_REF->SetDirectory(nullptr);
  Edep_REF->SetDirectory(nullptr);
  file->Close();
  //delete file;

  //return PSL_REF;
  return Edep_REF;

}


void Calcul_Chi2(TH1D* O, TH1D* C)
{
  Chi2=0;
  int n = C->GetNbinsX();
  //  cout << "n = " << n << endl;

  //O->Scale(C->Integral()/O->Integral());

  float sigma=0;

  for (int i=1; i<=n; i++)
    {
      sigma = C->GetBinError(i);
      //Chi2+= ((O->GetBinContent(i) - C->GetBinContent(i)) * (O->GetBinContent(i) - C->GetBinContent(i)) / (C->GetBinContent(i)));
      if(C->GetBinContent(i) !=0 && sigma !=0)
	{
	  Chi2+= ((O->GetBinContent(i) - C->GetBinContent(i)) * (O->GetBinContent(i) - C->GetBinContent(i))) / (C->GetBinContent(i) + (sigma*sigma));

	}
    }

}



Double_t fitFunc(float x, Double_t* par)
{
  Double_t PDF =0.0;
  PDF = 10*par[0]*exp(-(x*10)/par[1]);

  return PDF;

}


void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  PSL_Rec->Reset();
  
  for (int i=1; i<=100; i++)
    {
      for (int j=1; j<=15; j++)
	{
	  PSL_Rec->SetBinContent(j, PSL_Rec->GetBinContent(j) + fitFunc(i-0.5, par)*Matrice->GetBinContent(j, i));
	}
    }



  Calcul_Chi2(PSL_Rec, Data);
  f = Chi2;

}



void FIT_MINUIT_G()
{
  gStyle->SetOptStat(0);

  TMinuit *gMinuit = new TMinuit(2);  //initialize TMinuit with a maximum of 4 params
  gMinuit->SetFCN(fcn);

  Double_t arglist[10];
  Int_t ierflg = 0;

  //gMinuit->SetPrintLevel(-1);
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

  // Set starting values and step sizes for parameters
  gMinuit->mnparm(0, "A", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(1, "E0", start_E, step_E, low_E, up_E, ierflg);

  //gMinuit->FixParameter(0);
  //gMinuit->FixParameter(1);

  // Now ready for minimization step
  arglist[0] = 500;
  arglist[1] = 1.;

  //  gMinuit->mnexcm("CALL FCN", arglist ,1,ierflg);
  gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);
  gMinuit->mnexcm("HESSE", arglist ,2,ierflg);

  // Print results
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);

  h_g = (TH1D*)PSL_Rec->Clone("h_g");
  h_g->Draw("");
  //h_g->GetYaxis()->SetRangeUser(1, 50000);
  Data->Draw("PEsame");
  Data->SetLineColor(kBlack);
  h_g->SetLineColor(kRed);
  h_g->SetLineWidth(3);
  Data->SetLineWidth(3);
  h_g->GetXaxis()->SetTitle("IP number");
  h_g->GetYaxis()->SetTitle("PSL/mm^{2}");
  h_g->GetYaxis()->SetLabelSize(0.02);

  gMinuit->GetParameter(0, A, err_A);
  gMinuit->GetParameter(1, E0, err_E0);


  float Max = h_g->GetMaximum();

  TLatex *lData = new TLatex(10, 10000, "Data");
  lData->Draw();
  lData->SetTextSize(0.04);
  lData->SetTextColor(kBlack);

  TLatex *lFit = new TLatex(10, 1000, "Fit 1 Temperature");
  lFit->Draw();
  lFit->SetTextSize(0.04);
  lFit->SetTextColor(kRed);


  cout << "\n\n#######################################################" << endl;
  cout << "################## RESULTS OF FIT #####################" << endl;
  cout << "#######################################################" << endl;
  cout << "\nN0 = " << A << " +- " << err_A << endl;
  cout << "E0 = " << E0 << " +- " << err_E0 << " MeV" <<  endl;
  cout << "Chi2 = " << Chi2 << endl;
  cout << "\n\n" << endl;
}




Double_t fitFunc2T(float x, Double_t* par)
{
  Double_t PDF =0.0;
  PDF = 10*par[0]*exp(-(x*10)/par[1]) +10*par[2]*exp(-(x*10)/par[3]);
  // cout << "x = " << x << endl;
  // cout << "par 0 =" << par[0] << endl;
  // cout << "par 1 =" << par[1] << endl;
  // cout << "PDF = " << PDF << endl;

  return PDF;

}


void fcn2T(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  PSL_Rec->Reset();
  
  for (int i=1; i<=100; i++)
    {
      for (int j=1; j<=15; j++)
	{
	  PSL_Rec->SetBinContent(j, PSL_Rec->GetBinContent(j) + fitFunc2T(i-0.5, par)*Matrice->GetBinContent(j, i));
	}
    }



  Calcul_Chi2(PSL_Rec, Data);
  f = Chi2;
  //cout << "A = " << par[0] << endl;
  //cout << "E0 = " << par[1] << endl;
  //cout << "Chi2 = " << Chi2 << endl;

}



void FIT_MINUIT_G_2T()
{
  TMinuit *gMinuit = new TMinuit(4);  //initialize TMinuit with a maximum of 4 params
  gMinuit->SetFCN(fcn2T);

  Double_t arglist[10];
  Int_t ierflg = 0;
  
  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);
  
  // Set starting values and step sizes for parameters
  gMinuit->mnparm(0, "Acold", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(1, "Ecold", start_Ecold, step_Ecold, low_Ecold, up_Ecold, ierflg);
  gMinuit->mnparm(2, "Ahot", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(3, "Ehot", start_Ehot, step_Ehot, low_Ehot, up_Ehot, ierflg);
  
  // Now ready for minimization step
  arglist[0] = 1000;
  arglist[1] = 1.;

  //  gMinuit->mnexcm("CALL FCN", arglist ,1,ierflg);
  gMinuit->mnexcm("MIGRAD", arglist ,4,ierflg);
  
  // Print results
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);


  h_g_2T = (TH1D*)PSL_Rec->Clone("h_g_2T");
  h_g_2T->Draw("PEsame");
  //h_g_2T->GetYaxis()->SetRangeUser(1, 50000);
  Data->Draw("PEsame");
  Data->SetLineColor(kBlack);
  h_g_2T->SetLineColor(kCyan);
  h_g_2T->SetLineWidth(3);
  Data->SetLineWidth(3);
  h_g_2T->GetXaxis()->SetTitle("IP number");
  h_g_2T->GetYaxis()->SetTitle("PSL/mm^{2}");
  h_g_2T->GetYaxis()->SetLabelSize(0.02);

  float Max = h_g->GetMaximum();

  TLatex *lData = new TLatex(10, 10000, "Data");
  lData->Draw();
  lData->SetTextSize(0.04);
  lData->SetTextColor(kBlack);

  TLatex *lFit = new TLatex(10, 100, "Fit 2 Temperatures");
  lFit->Draw();
  lFit->SetTextSize(0.04);
  lFit->SetTextColor(kCyan);

  gMinuit->GetParameter(0, Acold, err_Acold);
  gMinuit->GetParameter(1, Ecold, err_Ecold);
  gMinuit->GetParameter(2, Ahot, err_Ahot);
  gMinuit->GetParameter(3, Ehot, err_Ehot);
  cout << "Acold = " << Acold << " +- " << err_Acold << endl;
  cout << "Ecold = " << Ecold << " +- " << err_Ecold << endl;
  cout << "Ahot = " << Ahot << " +- " << err_Ahot << endl;
  cout << "Ehot = " << Ehot << " +- " << err_Ehot << endl;
  cout << "N = " << (Acold*Ecold+Ahot*Ehot) << " +- " << (Ecold*err_Acold + Acold*err_Ecold + Ehot*err_Ahot + Ahot*err_Ehot) << endl;
  cout << "Chi2 = " << Chi2 << endl;
  cout << "Fit status = " << gMinuit->GetStatus() << endl;
}




void Histo_DATA(float IP[])
{
  for (int i=0; i<15; i++)
  {
    Data->SetBinContent(i+1, IP[i]);
  }
}



void Draw_Incident_Spectrum_Fit(float solid_angle)
{

  string seN, seNN;
  float Eval =0;

  A = A/solid_angle;
  err_A = err_A/solid_angle;

  Acold = Acold/solid_angle;
  err_Acold = err_Acold/solid_angle;

  Ahot = Ahot/solid_angle;
  err_Ahot = err_Ahot/solid_angle;

  f1 = new TF1("f1", "[0]*exp(-x/[1])", 0, 1000);
  f1->SetParameter(0, A);
  f1->SetParameter(1, E0);

  TF1* f2 = new TF1("f2", "[0]*exp(-x/[1]) + [2]*exp(-x/[3])", 0, 1000);
  f2->SetParameter(0, Acold);
  f2->SetParameter(1, Ecold);
  f2->SetParameter(2, Ahot);
  f2->SetParameter(3, Ehot);
  if (Chi2_gamma_store <= Chi2_gamma_2T_store) Eval = f1->Integral(0, 1000);
  if (Chi2_gamma_2T_store < Chi2_gamma_store) Eval = f2->Integral(0, 1000);
  
  f1->SetNpx(200);
  f1->Draw("");
  f1->SetLineColor(kRed);
  f1->SetLineWidth(3);

  f2->SetNpx(200);
  f2->Draw("same");
  f2->SetLineColor(kBlue);
  f2->SetLineWidth(2);




  
  float Max = f1->GetMaximum();
  f1->GetYaxis()->SetRangeUser(1, 10*Max);
  N = f1->Integral(0, 1000);
  eN = (N*(err_E0/E0 + err_A/A));
  
  seNN = Form("+-  %g #gamma/sr", eN);

  
  string sE = Form("Eo =  %g ", E0);
  TLatex *lE = new TLatex(300, 1*Max, sE.c_str());
  lE->Draw();
  lE->SetTextSize(0.035);
  lE->SetTextColor(kRed);

  string seE = Form("+-  %g keV", err_E0);
  TLatex *leE = new TLatex(600, 1*Max, seE.c_str());
  leE->Draw();
  leE->SetTextSize(0.035);
  leE->SetTextColor(kRed);

  string sNN = Form("N =  %g ", N);
  TLatex *lNN = new TLatex(300, 0.5*Max, sNN.c_str());
  lNN->Draw();
  lNN->SetTextSize(0.035);
  lNN->SetTextColor(kRed);

  TLatex *leNN = new TLatex(600, 0.5*Max, seNN.c_str());
  leNN->Draw();
  leNN->SetTextSize(0.035);
  leNN->SetTextColor(kRed);

  string sChi = Form("Chi2 =  %g ", Chi2_gamma_store);
  TLatex *lChi = new TLatex(300, 0.2*Max, sChi.c_str());
  lChi->Draw();
  lChi->SetTextSize(0.035);
  lChi->SetTextColor(kRed);

  f1->GetYaxis()->SetTitle("dN/dE [keV^{-1} sr^{-1}]");
  f1->GetXaxis()->SetTitle("Energy [keV]");


  string sEcold= Form("E_{cold} =  %g ", Ecold);
  TLatex *lEcold = new TLatex(300,0.01*Max, sEcold.c_str());
  lEcold->Draw();
  lEcold->SetTextSize(0.035);
  lEcold->SetTextColor(kBlue);

  string seEcold = Form("+-  %g keV", err_Ecold);
  TLatex *leEcold = new TLatex(600, 0.01*Max, seEcold.c_str());
  leEcold->Draw();
  leEcold->SetTextSize(0.035);
  leEcold->SetTextColor(kBlue);

  string sEhot= Form("E_{hot} =  %g ", Ehot);
  TLatex *lEhot = new TLatex(300,0.005*Max, sEhot.c_str());
  lEhot->Draw();
  lEhot->SetTextSize(0.035);
  lEhot->SetTextColor(kBlue);

  string seEhot = Form("+-  %g keV", err_Ehot);
  TLatex *leEhot = new TLatex(600, 0.005*Max, seEhot.c_str());
  leEhot->Draw();
  leEhot->SetTextSize(0.035);
  leEhot->SetTextColor(kBlue);
  
  float N2 = f2->Integral(0, 1000);
  float eN2 = (N2*(err_Ecold/Ecold + err_Ehot/Ehot + err_Acold/Acold + err_Ahot/Ahot));


  string sNN2 = Form("N =  %g ", N2);
  TLatex *lNN2 = new TLatex(300, 0.002*Max, sNN2.c_str());
  lNN2->Draw();
  lNN2->SetTextSize(0.035);
  lNN2->SetTextColor(kBlue);

  string seNN2 = Form("+-  %g #gamma/sr", eN2);
  TLatex *leNN2 = new TLatex(600, 0.002*Max, seNN2.c_str());
  leNN2->Draw();
  leNN2->SetTextSize(0.035);
  leNN2->SetTextColor(kBlue);

  string sChi2 = Form("Chi2 =  %g ", Chi2_gamma_2T_store);
  TLatex *lChi2 = new TLatex(300, 0.0008*Max, sChi2.c_str());
  lChi2->Draw();
  lChi2->SetTextSize(0.035);
  lChi2->SetTextColor(kBlue);

}



void Lecture_analyse_file(const char* filename)
{
  ifstream file(filename);
  string junk1, junk2, junk3;
  float Surface_pixel=0;
  
  for (int i=0; i<=15; i++)
    {
      if (i==0)
	{
	  file >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 >> junk1 ;
	  //cout << "junk = " << junk1 << endl;
	}
      
      if (i >0)
	{
	  //file >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> QL[i-1] >> junk3 >> junk3;
	  file >> junk2 >> junk2 >> junk2 >> junk2 >> junk2 >> Surface_pixel >> junk2 >> junk2 >> junk2 >> QL[i-1] >> junk3 >> junk3;
	  //cout << "junk2 = " << junk2 << endl;
	  cout << "QL[" << i-1<< "] = " << QL[i-1] << endl;
	  cout << "Surface pixel = " << Surface_pixel << endl;
	  IP[i-1] = QL_to_PSL(QL[i-1])/(Surface_pixel*0.0025);
	  cout << "IP[" << i-1<< "] = " << IP[i-1] << endl;
	}
      
    }

  cout << "SURFACE = " << Surface_pixel*0.0025 << " mm²" << endl;
  
}



void Routine_Analyse(const char* filename, const char* name, float Coef_PSL, float solid_angle, double Surface)
{
  
  cout << "\n\nfilename = " << filename << endl;
  cout << "PSL Coef = " << Coef_PSL << endl;
  cout << "Solid angle = " << solid_angle << endl;
  cout << "Surface = " << Surface << endl;

  Init_IP();
  Lecture_analyse_file(filename);
  //Test1();

  //MATRICE GAMMA PART 1 TEMP

  DefineMatrice(Coef_PSL, Surface);

  Create_Pad_Canvas("RESULTS GAMMA");
  Histo_DATA(IP);
  pad1->cd();
  pad1->SetLogy();
  FIT_MINUIT_G();
  Chi2_gamma_store=Chi2;
  FIT_MINUIT_G_2T();
  Chi2_gamma_2T_store=Chi2;


  
  pad2->cd();
  pad2->SetLogy();
  Draw_Incident_Spectrum_Fit(solid_angle);

  string fname = name;
  string part = "_Gamma.png";
  string pngname1 = fname+part;
  //cout << "png name = " << pngname1 << endl;

  c->SaveAs(pngname1.c_str());

  cout << "\nGAMMA RESULTS : " << endl;
  cout << "Chi2 1T = " << Chi2_gamma_store << endl;
  cout << "Chi2 2T = " << Chi2_gamma_2T_store << endl;


}










void Analyse()
{

  Init_IP();
  
  // cout << "\n\n1/ Measurement with electrons [1] or only with gammas [0] ?" << endl;
  // cin >> flag_electron;
  // if(flag_electron ==0) cout << "Measurement only with gammas !!!!" << endl;
  // if(flag_electron ==1) cout << "Measurement with electrons !!!!" << endl;
  // if(flag_electron >1)

  
  // if(flag_electron <=1)
  //   {
    cout << "\n\n2/ Enter the PSL coefficient [PSL/MeV] (default is 6.95E-1) :" << endl;
    cin >> Coef_PSL;

    cout << "\n\nEnter the solid angle :" << endl;
    cin >> solid_angle;

    //Programme_Test();

    cout << "Enter the number of PSL/mm² in IP number 1 :" << endl;
    cin >> IP[0];

    cout << "Enter the number of PSL/mm² in IP number 2 :" << endl;
    cin >> IP[1];

    cout << "Enter the number of PSL/mm² in IP number 3 :" << endl;
    cin >> IP[2];

    cout << "Enter the number of PSL/mm² in IP number 4 :" << endl;
    cin >> IP[3];

    cout << "Enter the number of PSL/mm² in IP number 5 :" << endl;
    cin >> IP[4];

    cout << "Enter the number of PSL/mm² in IP number 6 :" << endl;
    cin >> IP[5];

    cout << "Enter the number of PSL/mm² in IP number 7 :" << endl;
    cin >> IP[6];

    //  }

  //MATRICE GAMMA PART

  DefineMatrice(Coef_PSL, Surface);

  Create_Pad_Canvas("RESULTS GAMMA");
  Histo_DATA(IP);
  pad1->cd();
  pad1->SetLogy();
  FIT_MINUIT_G();
  Chi2_gamma_store=Chi2;

  pad2->cd();
  pad2->SetLogy();
  Draw_Incident_Spectrum_Fit(solid_angle);


  //ELECTRONS GAMMA PART

  Create_Pad_Canvas("RESULTS ELECTRONS");
  pad1->cd();
  pad1->SetLogy();
  DefineMatriceElectron(Coef_PSL, Surface);


  pad2->cd();
  pad2->SetLogy();
  Draw_Incident_Spectrum_Fit(solid_angle);


  cout << "\nGAMMA RESULTS : " << endl;
  cout << "Chi2 gen = " << Chi2_gamma_store << endl;

  cout << "\nELECTRONS RESULTS : " << endl;
  cout << "Chi2 gen = " << Chi2_store << endl;

}
