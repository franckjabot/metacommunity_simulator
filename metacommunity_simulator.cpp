/************************************************************************************
    Simulations of the temporal dynamics of metacommunities
	Franck Jabot, 14th October 2017.

Last modified on 20th June 2018.
	compilation: g++-3 -O3 -o metacommunity_simulator metacommunity_simulator.cpp -lgsl -lgslcblas -lm
*************************************************************************************/
// Libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <limits>
#include <string>
#include <math.h>
#include <malloc.h>
#include <list>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
using namespace std; 

//Random number generator
const gsl_rng_type * T;
gsl_rng * r;

void initialize_environment(double **generated_environment,double *param_environment, int n_trait,int sqrt_n_patches,gsl_rng *r){
	if (param_environment[0]!=-1.0){
		int n_patches=sqrt_n_patches*sqrt_n_patches;
		double min_trend=0.5*(0.0-param_environment[0]);
		double max_trend=0.5*(0.0+param_environment[0]);
		double step_trend=(max_trend-min_trend)/(sqrt_n_patches-1.0);
		double global =  gsl_ran_flat(r,0.5*(0.0-param_environment[1]),0.5*(0.0+param_environment[1]));
		double ne;
		int i=0;
		for (int i1=0;i1<sqrt_n_patches;i1++){
			for (int i2=0;i2<sqrt_n_patches;i2++){
				i=(i1*sqrt_n_patches)+i2;
				for (int j=0;j<n_trait;j++){
					if (generated_environment[i][j]==0.0){
						generated_environment[i][j]=0.5+(min_trend+i1*step_trend)+global+gsl_ran_flat(r,0.5*(0.0-param_environment[2]),0.5*(0.0+param_environment[2]));
					}
					else{
						ne=0.5+(min_trend+i1*step_trend)+global+gsl_ran_flat(r,0.5*(0.0-param_environment[2]),0.5*(0.0+param_environment[2]));
						ne*=(1.0-param_environment[3]);
						ne+=(param_environment[3]*generated_environment[i][j]);
						generated_environment[i][j]=ne;
					}
				}
			}
		}
	}
}

void initialize_regional_pool(double *regional_pool,double *param_pool,int S,gsl_rng *r){
	if (param_pool[0]==1.0){
		for (int i=0;i<S;i++){
			regional_pool[i]=1.0/(S+0.0);
		}
	}
	if (param_pool[0]==2.0){
		double *alpha;
		alpha=new double[S];
		for (int i=0;i<S;i++){
			alpha[i]=param_pool[1];
		}
		gsl_ran_dirichlet(r,S,alpha,regional_pool);
	}
}

void initialize_metacom(unsigned int **metacom,double *regional_pool,int n_patches,int S,int iK_loc,gsl_rng *r){
	for (int i=0;i<n_patches;i++){
		gsl_ran_multinomial(r,S,iK_loc,regional_pool,metacom[i]);
		metacom[i][S]=iK_loc;
	}
}

void compute_fitness(double **table_fitness,int n_patches,int S,double **environment,double *sig, double A,int n_trait,double **species_traits){
	double dif=0.0;
	for (int i=0;i<n_patches;i++){
		for (int j=0;j<S;j++){
			table_fitness[i][j]=A;
			for (int k=0;k<n_trait;k++){
				dif=species_traits[j][k]-environment[i][k];
				table_fitness[i][j]*=exp(-(dif*dif)/(2*sig[k]*sig[k]));
			}
			table_fitness[i][j]+=1.0;
		}
	}
}

void dispersal(unsigned int **metacom,unsigned int **propagule_cloud,double **table_fitness,int n_patches,int sqrt_n_patches,int S,double d,double m,double K_loc,double LDD,double *regional_pool,gsl_rng *r){
	// Computation of propagule cloud
	unsigned int n_recruits=0;
	for (int i=0;i<n_patches;i++){
		for (int j=0;j<S;j++){
			propagule_cloud[i][j]=0;
		}
	}
	double J;
	for (int i=0;i<n_patches;i++){
		J=metacom[i][S];
		if (i>=1){
			for (int j=0;j<S;j++){
				propagule_cloud[(i-1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i-1)][j]); 
			}
			if (i>=(sqrt_n_patches-1)){
				for (int j=0;j<S;j++){
					propagule_cloud[(i-sqrt_n_patches+1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i-sqrt_n_patches+1)][j]); 
				}
				if (i>=sqrt_n_patches){
					for (int j=0;j<S;j++){
						propagule_cloud[(i-sqrt_n_patches)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i-sqrt_n_patches)][j]); 
					}
					if (i>sqrt_n_patches){
						for (int j=0;j<S;j++){
							propagule_cloud[(i-sqrt_n_patches-1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i-sqrt_n_patches-1)][j]); 
						}
					}
				}
			}
		}
		if (i<(n_patches-1)){
			for (int j=0;j<S;j++){
				propagule_cloud[(i+1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i+1)][j]); 
			}
			if (i<(n_patches-sqrt_n_patches+1)){
				for (int j=0;j<S;j++){
					propagule_cloud[(i+sqrt_n_patches-1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i+sqrt_n_patches-1)][j]); 
				}
				if (i<(n_patches-sqrt_n_patches)){
					for (int j=0;j<S;j++){
						propagule_cloud[(i+sqrt_n_patches)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i+sqrt_n_patches)][j]); 
					}
					if (i<(n_patches-sqrt_n_patches-1)){
						for (int j=0;j<S;j++){
							propagule_cloud[(i+sqrt_n_patches+1)][j]+=gsl_ran_poisson(r,d*m*0.125*metacom[i][j]*table_fitness[(i+sqrt_n_patches+1)][j]); 
						}
					}
				}
			}
		}
	}
	// Computation of local recruitment
	for (int i=0;i<n_patches;i++){
		J=metacom[i][S];
		for (int j=0;j<S;j++){
			propagule_cloud[i][j]+=(gsl_ran_poisson(r,d*(1-m)*metacom[i][j]*table_fitness[i][j])+gsl_ran_poisson(r,LDD*regional_pool[j]*table_fitness[i][j]));
		}
	}
}

void establishment(unsigned int **metacom,unsigned int **propagule_cloud,double *prop,int n_patches,int S,double K_loc,gsl_rng *r){
	//From neighboring patches + LDD
	unsigned int n_recruits=0;
	for (int i=0;i<n_patches;i++){
		if (metacom[i][S]<K_loc){
			n_recruits=gsl_ran_poisson(r,K_loc-metacom[i][S]);
			for (int j=0;j<S;j++){
				prop[j]=propagule_cloud[i][j]+0.0;
			}
			gsl_ran_multinomial(r,S,n_recruits,prop,propagule_cloud[i]);
			for (int j=0;j<S;j++){
				metacom[i][j]+=propagule_cloud[i][j];
				metacom[i][S]+=propagule_cloud[i][j];
			}
		}
	}
}

void mortality(unsigned int **metacom,unsigned int **deadcom,double **table_fitness,double A,double d,int n_patches,int S,double K_loc,gsl_rng *r){
	unsigned int ndead=0;
	int J=0;
	for (int i=0;i<n_patches;i++){
		J=metacom[i][S];
		deadcom[i][S]=0;
		for (int j=0;j<S;j++){
			if (metacom[i][j]>0){
				ndead = (metacom[i][j] - gsl_ran_binomial(r,(1.0-d)*(table_fitness[i][j]/(1.0+A)),metacom[i][j])); 
				if (ndead>metacom[i][j]){
					ndead=metacom[i][j];
				}
				deadcom[i][j]=ndead;
				deadcom[i][S]+=ndead; 		
			}
			else{
				deadcom[i][j]=0;
			}
		}
	}
}

void update_mortality(unsigned int **metacom,unsigned int **deadcom,int n_patches,int S){
	for (int i=0;i<n_patches;i++){
		for (int j=0;j<(S+1);j++){
		  	metacom[i][j]-=deadcom[i][j];
		}
	}
}

void step_dynamics(unsigned int **metacom,unsigned int **deadcom,unsigned int **propagule_cloud,double *regional_pool,double *prop,int n_patches,int sqrt_n_patches,int S,double K_loc,double d,double m,double LDD,double **table_fitness,double **environment,double *sig, double A,int n_trait,double **species_traits,gsl_rng *r){
	//fitness computing
	compute_fitness(table_fitness,n_patches,S,environment,sig,A,n_trait,species_traits);
	//mortality
	mortality(metacom,deadcom,table_fitness,A,d,n_patches,S,K_loc,r);
	//Reproduction and dispersal
	dispersal(metacom,propagule_cloud,table_fitness,n_patches,sqrt_n_patches,S,d,m,K_loc,LDD,regional_pool,r);
	//update mortality
	update_mortality(metacom,deadcom,n_patches,S);
	//Recruitment
	establishment(metacom,propagule_cloud,prop,n_patches,S,K_loc,r);
}

void compute_stats_previous(unsigned int **metacom,int n_patches,int S,double **stats){
	for (int i=0;i<n_patches;i++){
		stats[i][0]=0; // S
		stats[i][1]=metacom[i][S]*log(metacom[i][S]+0.0); // H
		for (int j=0;j<S;j++){
			if(metacom[i][j]>0){
				stats[i][0]+=1;
				stats[i][1]-=metacom[i][j]*log(metacom[i][j]+0.0);
			}
		}
		stats[i][1]/=(metacom[i][S]+0.0);
	}
}

void compute_stats(unsigned int **metacom,int n_patches,int S,double **stats){
	for (int i=0;i<n_patches;i++){
		stats[i][0]=0; // S
		for (int j=0;j<S;j++){
			if(metacom[i][j]>0){
				stats[i][0]+=1;
			}
		}
	}
}

void copy_pile(unsigned int **metaco,unsigned int ***metaco_pile,int istep,int n_patches,int S){
	for (int i=0;i<n_patches;i++){
		for (int j=0;j<(S+1);j++){
			metaco_pile[istep][i][j]=metaco[i][j];
		}
	}
}

void copy_pile_env(double **generated_env,double ***env_pile,int istep,int n_patches,int n_trait){
	for (int i=0;i<n_patches;i++){
		for (int j=0;j<n_trait;j++){
			env_pile[istep][i][j]=generated_env[i][j];
		}
	}
}

void compute_beta(unsigned int ***metacom_pile,double ***env_pile,int n_patches,int S,int n_trait,int nstep, int *sampled,int nsamp,int sqrt_n_patches,double **stats_beta){
	int ind=0;
	double alpha1=0;
	double alpha2=0;
	int x1,y1,x2,y2,s1,s2,s3,s4;
	double temp;
	for (int i1=0;i1<nsamp;i1++){
		x1=sampled[i1]/sqrt_n_patches;
		y1=sampled[i1]%sqrt_n_patches;
		for (int i2=i1;i2<nsamp;i2++){
			x2=sampled[i2]/sqrt_n_patches;
			y2=sampled[i2]%sqrt_n_patches;
			if (i2>i1){
				for (int j1=0;j1<nstep;j1++){
					alpha1=0;
					for (int i=0;i<S;i++){
						if (metacom_pile[j1][sampled[i1]][i]>0){
							alpha1+=1;
						}
					}
					for (int j2=0;j2<nstep;j2++){
						alpha2=0;
						for (int i=0;i<S;i++){
							if (metacom_pile[j2][sampled[i2]][i]>0){
								alpha2+=1;
							}
						}
						stats_beta[ind][0]=0.5*(metacom_pile[j1][sampled[i1]][S]+metacom_pile[j2][sampled[i2]][S]); // mean_J
						stats_beta[ind][1]=abs(metacom_pile[j1][sampled[i1]][S]-metacom_pile[j2][sampled[i2]][S]); // delta_J
						stats_beta[ind][2]=abs(alpha1-alpha2); // delta_alpha
						stats_beta[ind][3]=abs(j2-j1); // delta_t
						stats_beta[ind][4]=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)); // delta_x
						stats_beta[ind][5]=0;
						for (int i=0;i<n_trait;i++){
							stats_beta[ind][5]+=((env_pile[j1][sampled[i1]][i]-env_pile[j2][sampled[i2]][i])*(env_pile[j1][sampled[i1]][i]-env_pile[j2][sampled[i2]][i]));
						}
						temp=sqrt(stats_beta[ind][5]);
						stats_beta[ind][5]=temp; // delta_env
						stats_beta[ind][6]=0;
						s1=0;
						s2=0;
						s3=0;
						s4=0;
						for (int i=0;i<S;i++){
							if (metacom_pile[j1][sampled[i1]][i]>0){
								s1+=1;
							}
							if (metacom_pile[j2][sampled[i2]][i]>0){
								s2+=1;
							}
							if ((metacom_pile[j1][sampled[i1]][i]>0)&&(metacom_pile[j2][sampled[i2]][i]>0)){
								s3+=1;
								s4+=min(metacom_pile[j1][sampled[i1]][i],metacom_pile[j2][sampled[i2]][i]);
							}
						}
						stats_beta[ind][6]=1.0-((2*s3+0.0)/(s1+s2+0.0)); // beta_sor
						stats_beta[ind][7]=1.0-((2*s4+0.0)/(metacom_pile[j1][sampled[i1]][S]+metacom_pile[j2][sampled[i2]][S]+0.0)); //bray-curtis
						stats_beta[ind][8]=sampled[i1];
						stats_beta[ind][9]=j1;
						stats_beta[ind][10]=sampled[i2];
						stats_beta[ind][11]=j2;
						ind=ind+1;
					}
				}
			}
			else{
				for (int j1=0;j1<(nstep-1);j1++){
					alpha1=0;
					for (int i=0;i<S;i++){
						if (metacom_pile[j1][sampled[i1]][i]>0){
							alpha1+=1;
						}
					}
					for (int j2=(j1+1);j2<nstep;j2++){
						alpha2=0;
						for (int i=0;i<S;i++){
							if (metacom_pile[j2][sampled[i2]][i]>0){
								alpha2+=1;
							}
						}
						stats_beta[ind][0]=0.5*(metacom_pile[j1][sampled[i1]][S]+metacom_pile[j2][sampled[i2]][S]); // mean_J
						stats_beta[ind][1]=abs(metacom_pile[j1][sampled[i1]][S]-metacom_pile[j2][sampled[i2]][S]); // delta_J
						stats_beta[ind][2]=abs(alpha1-alpha2); // delta_alpha
						stats_beta[ind][3]=abs(j2-j1); // delta_t
						stats_beta[ind][4]=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)); // delta_x
						stats_beta[ind][5]=0;
						for (int i=0;i<n_trait;i++){
							stats_beta[ind][5]+=((env_pile[j1][sampled[i1]][i]-env_pile[j2][sampled[i2]][i])*(env_pile[j1][sampled[i1]][i]-env_pile[j2][sampled[i2]][i]));
						}
						temp=sqrt(stats_beta[ind][5]);
						stats_beta[ind][5]=temp; // delta_env
						stats_beta[ind][6]=0;
						s1=0;
						s2=0;
						s3=0;
						s4=0;
						for (int i=0;i<S;i++){
							if (metacom_pile[j1][sampled[i1]][i]>0){
								s1+=1;
							}
							if (metacom_pile[j2][sampled[i2]][i]>0){
								s2+=1;
							}
							if ((metacom_pile[j1][sampled[i1]][i]>0)&&(metacom_pile[j2][sampled[i2]][i]>0)){
								s3+=1;
								s4+=min(metacom_pile[j1][sampled[i1]][i],metacom_pile[j2][sampled[i2]][i]);
							}
						}
						stats_beta[ind][6]=1.0-((2*s3+0.0)/(s1+s2+0.0)); // beta_sor
						stats_beta[ind][7]=1.0-((2*s4+0.0)/(metacom_pile[j1][sampled[i1]][S]+metacom_pile[j2][sampled[i2]][S]+0.0)); //bray-curtis
						stats_beta[ind][8]=sampled[i1];
						stats_beta[ind][9]=j1;
						stats_beta[ind][10]=sampled[i2];
						stats_beta[ind][11]=j2;
						ind=ind+1;
					}
				}
			}
		}
	}
}
	

int main(){
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	unsigned long int seed = 2018;
    gsl_rng_set (r, seed);
	double *param_env;
	param_env=new double[4];
	param_env[0]=0.1;
	param_env[1]=0.1;
	param_env[2]=0.1;
	param_env[3]=0.0;
	int n_trait=1;
	int sqrt_n_patches=20;
	int n_patches=sqrt_n_patches*sqrt_n_patches;
	double **generated_env;
	generated_env= new double *[n_patches];
	for (int i=0;i<n_patches;i++){
		generated_env[i]=new double[n_trait];
		for (int j=0;j<n_trait;j++){
			generated_env[i][j]=0.0;
		}
	}
	int S=100;
	unsigned int **metaco;
	metaco=new unsigned int*[n_patches];
	for (int i=0;i<n_patches;i++){
		metaco[i]=new unsigned int[(S+1)];
		for (int j=0;j<(S+1);j++){
			metaco[i][j]=0;
		}
	}
	unsigned int **deadco;
	deadco=new unsigned int*[n_patches];
	for (int i=0;i<n_patches;i++){
		deadco[i]=new unsigned int[(S+1)];
		for (int j=0;j<(S+1);j++){
			deadco[i][j]=0;
		}
	}
	int iK=500;
	double K_loc=iK+0.0;
	double *regional_pool;
	regional_pool=new double[S];
	for (int i=0;i<S;i++){
		regional_pool[i]=0.01;
	}
	double *prop;
	prop=new double[S];
	for (int i=0;i<S;i++){
		prop[i]=0.0;
	}

	double **tab_fit;
	tab_fit=new double*[n_patches];
	for (int i=0;i<n_patches;i++){
		tab_fit[i]=new double[S];
		for (int j=0;j<S;j++){
			tab_fit[i][j]=0.0;
		}
	}
	double A=1000.0;
	double *sig;
	sig=new double[n_trait];
	for (int i=0;i<n_trait;i++){
		sig[i]=0.06;
	}
	double **species_traits;
	species_traits=new double*[S];
	for (int i=0;i<S;i++){
		species_traits[i]=new double[n_trait];
		for (int j=0;j<n_trait;j++){
			species_traits[i][j]=(i+0.0)/100.0;
		}
	}

	unsigned int **propagule_cloud;
	propagule_cloud= new unsigned int*[n_patches];
	for (int i=0;i<n_patches;i++){
		propagule_cloud[i]=new unsigned int[S];
		for (int j=0;j<S;j++){
			propagule_cloud[i][j]=0;
		}
	}
	double d=0.2;
	double m=0.1;
	double LDD=10.0;

	double *param_pool;
	param_pool=new double[2];
	param_pool[0]=1.0; // equal species abundances in the regional pool
	param_pool[1]=1.0;
	
	double **stats;
	int nstats=1;
	stats=new double*[n_patches];
	for (int i=0;i<n_patches;i++){
		stats[i]=new double[nstats];
		for (int j=0;j<nstats;j++){
			stats[i][j]=0;
		}
	}
	
	int nstep=20;
	unsigned int ***metaco_pile;
	metaco_pile=new unsigned int**[nstep];
	for (int k=0;k<nstep;k++){
		metaco_pile[k]=new unsigned int*[n_patches];
		for (int i=0;i<n_patches;i++){
			metaco_pile[k][i]=new unsigned int[(S+1)];
			for (int j=0;j<(S+1);j++){
				metaco_pile[k][i][j]=0;
			}
		}
	}
	double ***env_pile;
	env_pile=new double **[nstep];
	for (int k=0;k<nstep;k++){
		env_pile[k]= new double *[n_patches];
		for (int i=0;i<n_patches;i++){
			env_pile[k][i]=new double[n_trait];
			for (int j=0;j<n_trait;j++){
				env_pile[k][i][j]=0.0;
			}
		}
	}
	int nsamp=50;
	int *sampled;
	sampled = new int[nsamp];
	for (int i=0;i<nsamp;i++){
		sampled[i]=0;
	}
	double **stats_beta;
	int ns=(nsamp*nstep)*(nsamp*nstep-1)/2;
	stats_beta=new double*[ns];
	for (int i=0;i<ns;i++){
		stats_beta[i]=new double[12];
		for (int j=0;j<12;j++){
			stats_beta[i][j]=0;
		}
	}
	
	
	// OUTPUT
	char nomfo[256];
	sprintf(nomfo,"burn-in_S3_d%d_J%d_m%d_ldd%d_A%d_sig%d_env%d_%d_%d_%d.txt",int(d*10),iK,int(m*100),int(10*LDD),int(10*A),int(sig[0]*1000),int(param_env[0]*100),int(param_env[1]*100),int(param_env[2]*100),int(param_env[3]*100));
	ofstream outS(nomfo);
	
	// test of the necessary burn-in length
	// initialisation
	initialize_environment(generated_env,param_env,n_trait,sqrt_n_patches,r);
	initialize_regional_pool(regional_pool,param_pool,S,r);
	initialize_metacom(metaco,regional_pool,n_patches,S,iK,r);
	
	//burn-in
	for (int istep=0;istep<2500;istep++){
		initialize_environment(generated_env,param_env,n_trait,sqrt_n_patches,r);
		step_dynamics(metaco,deadco,propagule_cloud,regional_pool,prop,n_patches,sqrt_n_patches,S,K_loc,d,m,LDD,tab_fit,generated_env,sig,A,n_trait,species_traits,r);
		compute_stats(metaco,n_patches,S,stats);
		for (int i=0;i<n_patches;i++){
			outS<<stats[i][0];
			outS<<" ";
		}
		outS<<endl;
	}
	outS.flush();
    outS.close();
	
	sprintf(nomfo,"metacom_d%d_J%d_m%d_ldd%d_A%d_sig%d_env%d_%d_%d_%d.txt",int(d*10),iK,int(m*100),int(10*LDD),int(10*A),int(sig[0]*1000),int(param_env[0]*100),int(param_env[1]*100),int(param_env[2]*100),int(param_env[3]*100));
	ofstream outc(nomfo);
	sprintf(nomfo,"environment_d%d_J%d_m%d_ldd%d_A%d_sig%d_env%d_%d_%d_%d.txt",int(d*10),iK,int(m*100),int(10*LDD),int(10*A),int(sig[0]*1000),int(param_env[0]*100),int(param_env[1]*100),int(param_env[2]*100),int(param_env[3]*100));
	ofstream oute(nomfo);
	sprintf(nomfo,"S_d%d_J%d_m%d_ldd%d_A%d_sig%d_env%d_%d_%d_%d.txt",int(d*10),iK,int(m*100),int(10*LDD),int(10*A),int(sig[0]*1000),int(param_env[0]*100),int(param_env[1]*100),int(param_env[2]*100),int(param_env[3]*100));
	ofstream outS2(nomfo);
	for (int istep=0;istep<nstep;istep++){
		initialize_environment(generated_env,param_env,n_trait,sqrt_n_patches,r);
		copy_pile_env(generated_env,env_pile,istep,n_patches,n_trait);
		step_dynamics(metaco,deadco,propagule_cloud,regional_pool,prop,n_patches,sqrt_n_patches,S,K_loc,d,m,LDD,tab_fit,generated_env,sig,A,n_trait,species_traits,r);
		copy_pile(metaco,metaco_pile,istep,n_patches,S);
		compute_stats(metaco,n_patches,S,stats);
		for (int i=0;i<n_patches;i++){
			outS2<<stats[i][0]<<" ";
			for (int j=0;j<(S+1);j++){
				outc<<metaco[i][j]<<" ";
			}
			outc<<endl;
		}
		outS2<<endl;
		for (int i=0;i<n_patches;i++){
			for (int j=0;j<n_trait;j++){
				oute<<generated_env[i][j]<<" ";
			}
			oute<<endl;
		}
	}
	outS2.flush();
    outS2.close();
	oute.flush();
    oute.close();
	outc.flush();
    outc.close();
	
	int *b;
	b=new int[n_patches];
	for (int i=0;i<n_patches;i++){
		b[i]=i;
	}
	gsl_ran_choose(r,sampled,nsamp,b,n_patches,sizeof (int));
	compute_beta(metaco_pile,env_pile,n_patches,S,n_trait,nstep,sampled,nsamp,sqrt_n_patches,stats_beta);
	
	sprintf(nomfo,"beta_d%d_J%d_m%d_ldd%d_A%d_sig%d_env%d_%d_%d_%d.txt",int(d*10),iK,int(m*100),int(10*LDD),int(10*A),int(sig[0]*1000),int(param_env[0]*100),int(param_env[1]*100),int(param_env[2]*100),int(param_env[3]*100));
	ofstream outb(nomfo);
	for (int i=0;i<ns;i++){
		for (int j=0;j<12;j++){
			outb<<stats_beta[i][j]<<" ";
		}
		outb<<endl;
	}
	outb.flush();
    outb.close();
	
	return 0;
}
