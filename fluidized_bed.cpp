/*  Lattice Boltzmann sample, written in C++, using the OpenLB
 *  library
 *
 *  Copyright (C) 2024 Adrian Kummerlaender
 *  E-mail contact: info@openlb.net
 *  The most recent release of OpenLB can be downloaded at
 *  <http://www.openlb.net/>
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this program; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA  02110-1301, USA.
 */
#include <olb.h>
/* 
 * These are all the lammps header files we need 
 * */
#include "lammps/lammps.h"
#include "lammps/input.h"
#include "lammps/output.h"
#include "lammps/neighbor.h"
#include "lammps/neigh_list.h"
#include "lammps/atom.h"
#include "lammps/comm.h"
#include "lammps/error.h"
#include "lammps/fix.h"
#include "lammps/force.h"
#include "lammps/memory.h"
#include "lammps/modify.h"
#include "lammps/compute.h"
#include "lammps/variable.h"
#include "lammps/library.h"

#include <chrono>

using namespace olb;

using T = float;
using DESCRIPTOR = descriptors::D3Q19<
descriptors::POROSITY,
	descriptors::VELOCITY,
	fields::fsi::ELEMENT_TAG,
	fields::fsi::ELEMENT_FORCE
	>;

using BulkDynamics = dynamics::Tuple<
T, DESCRIPTOR,
momenta::BulkTuple,
equilibria::SecondOrder,
collision::BGK,
forcing::fsi::HLBM
>;

const T maxPhysT = 40;

//const T sphereA[nSpheres] {0.05, 0.1, 0.15};
//const T sphereF[nSpheres] {-3, 2.5, -2};

std::shared_ptr<IndicatorF3D<T>> makeDomainI() {
	std::shared_ptr<IndicatorF3D<T>> boxI(new IndicatorCuboid3D<T>({1,1,3},{0,0,0}));
	return boxI;
}

void prepareGeometry(const UnitConverter<T,DESCRIPTOR>& converter,
		SuperGeometry<T,3>& sGeometry)
{
	OstreamManager clout(std::cout, "prepareGeometry");
	clout << "Prepare Geometry ..." << std::endl;

	auto domainI = makeDomainI();

	sGeometry.rename(0, 2);
	//sGeometry.rename(2, 1, domainI);

	Vector<T,3> origin = sGeometry.getStatistics().getMinPhysR( 2 );
	//origin[1] += converter.getPhysDeltaX()/2.;
	//origin[2] += converter.getPhysDeltaX()/2.;

	Vector<T,3> extend = sGeometry.getStatistics().getMaxPhysR( 2 );
	extend[0] = extend[0]+converter.getPhysDeltaX();
	extend[1] = extend[1]+converter.getPhysDeltaX();

	// Set material number for inflow
	//origin[2] = sGeometry.getStatistics().getMinPhysR( 2 )[2]-converter.getPhysDeltaX();
	extend[2] = converter.getPhysDeltaX()/2.;
	IndicatorCuboid3D<T> inflow( extend,origin );
	sGeometry.rename( 2,3,inflow );

	// Set material number for outflow
	origin[2] = sGeometry.getStatistics().getMaxPhysR( 2 )[2]-converter.getPhysDeltaX();
	IndicatorCuboid3D<T> outflow( extend,origin );
	sGeometry.rename( 2,4,outflow );
	sGeometry.rename(2, 1);//, domainI);
	sGeometry.clean(true, {1});
	sGeometry.checkForErrors();
	sGeometry.print();

	clout << "Prepare Geometry ... OK" << std::endl;
}

void prepareLattice(SuperLattice<T,DESCRIPTOR>& sLattice,
		const UnitConverter<T,DESCRIPTOR>& converter,
		SuperGeometry<T,3>& sGeometry)
{
	OstreamManager clout( std::cout,"prepareLattice" );
	clout << "Prepare Lattice ..." << std::endl;

	sLattice.defineDynamics<NoDynamics>(sGeometry, 0);
	sLattice.defineDynamics<BulkDynamics>(sGeometry, 1);

	boundary::set<boundary::BounceBack>(sLattice, sGeometry, 2);

	boundary::set<boundary::InterpolatedVelocity>(sLattice, sGeometry, 3);
	boundary::set<boundary::InterpolatedPressure>(sLattice, sGeometry, 4);
	sLattice.setParameter<descriptors::OMEGA>(converter.getLatticeRelaxationFrequency());

	AnalyticalConst3D<T,T> rhoF(1);
	Vector<T,3> velocityV;
	AnalyticalConst3D<T,T> uF(velocityV);

	auto bulkIndicator = sGeometry.getMaterialIndicator({1});
	sLattice.iniEquilibrium(bulkIndicator, rhoF, uF);
	sLattice.defineRhoU(bulkIndicator, rhoF, uF);

	{
		AnalyticalConst3D<T,T> porosityF(1);
		sLattice.defineField<descriptors::POROSITY>(bulkIndicator, porosityF);
	}

	sLattice.initialize();

	clout << "Prepare Lattice ... OK" << std::endl;
}

void getResults(SuperLattice<T,DESCRIPTOR>& sLattice,
		const UnitConverter<T,DESCRIPTOR>& converter,
		SuperGeometry<T,3>& sGeometry,
		util::Timer<T>& timer,
		std::size_t iT)
{
	OstreamManager clout(std::cout, "getResults");

	const int vtkIter  = converter.getLatticeTime(0.1);
	const int statIter = converter.getLatticeTime(0.001);

	if (iT == 0) {
		SuperVTMwriter3D<T> vtmWriter("flow");
		vtmWriter.createMasterFile();
	}

	if (iT % statIter == 0) {
		timer.update(iT);
		timer.printStep();
		sLattice.getStatistics().print(iT, converter.getPhysTime(iT));
		if (std::isnan(sLattice.getStatistics().getAverageRho())) {
			std::exit(-1);
		}
	}

	if (iT % vtkIter == 0) {
		sLattice.executePostProcessors(stage::Evaluation{});
		sLattice.setProcessingContext(ProcessingContext::Evaluation);
		sLattice.scheduleBackgroundOutputVTK([&,iT](auto task) {
				SuperVTMwriter3D<T> vtkWriter("flow");
				SuperLatticePhysVelocity3D velocity(sLattice, converter);
				SuperLatticeField3D<T,DESCRIPTOR,descriptors::POROSITY> porosity(sLattice);
				//SuperLatticeField3D<T,DESCRIPTOR,fields::fsi::ELEMENT_TAG> tag(sLattice);
				//SuperLatticeField3D<T,DESCRIPTOR,fields::fsi::ELEMENT_FORCE> force(sLattice);
				vtkWriter.addFunctor(velocity);
				vtkWriter.addFunctor(porosity);
				//vtkWriter.addFunctor(tag);
				//vtkWriter.addFunctor(force);
				vtkWriter.write(iT);
				});
	}
}



// Generates a slowly increasing inflow for the first iTMaxStart timesteps
void setBoundaryValues( SuperLattice<T, DESCRIPTOR>& sLattice,
                        UnitConverter<T,DESCRIPTOR> const& converter, int iT,
                        SuperGeometry<T,3>& superGeometry )
{
  OstreamManager clout( std::cout,"setBoundaryValues" );

  // No of time steps for smooth start-up
  int iTmaxStart = converter.getLatticeTime( maxPhysT*0.2 );
  int iTupdate = 10;

  if ( iT%iTupdate == 0 && iT <= iTmaxStart ) {
    // Smooth start curve, sinus
    // SinusStartScale<T,int> StartScale(iTmaxStart, T(1));

    // Smooth start curve, polynomial
    PolynomialStartScale<T,int> StartScale( iTmaxStart, T( 1 ) );

    // Creates and sets the Poiseuille inflow profile using functors
    int iTvec[1] = {iT};
    T frac[1] = {};
    StartScale( frac,iTvec );
    std::vector<T> maxVelocity( 3,0 );
    maxVelocity[2] = 0.25*frac[0]*converter.getCharLatticeVelocity();

    T distance2Wall = converter.getPhysDeltaX()/2.;
    RectanglePoiseuille3D<T> poiseuilleU( superGeometry, 3, maxVelocity, distance2Wall, distance2Wall, distance2Wall );
    sLattice.defineU( superGeometry, 3, poiseuilleU );

    clout << "step=" << iT << "; maxVel=" << maxVelocity[2] << std::endl;

    sLattice.setProcessingContext<Array<momenta::FixedVelocityMomentumGeneric::VELOCITY>>(
      ProcessingContext::Simulation);
  }
}


void getGlobalPositionVelocity(LAMMPS_NS::LAMMPS *lmp,double *globalPositions, double *globalVelocities, double *globalRadii, int *reverseTags)
{

	int nAtoms = lmp->atom->natoms;
	double **x = lmp->atom->x;
	double **v = lmp->atom->v;
	double *r = lmp->atom->radius;
	int nLocalAtomsVec=lmp->atom->nlocal*3; // This the number of atoms owned by this processor. 
	int nLocalAtoms=lmp->atom->nlocal; // This the number of atoms owned by this processor. 
	double * localPositions = new double [nLocalAtomsVec];
	double * localVelocities = new double [nLocalAtomsVec];
	double * localRadii = new double [nLocalAtoms];
	int * localTags = new int [nLocalAtoms];
	for(int localIndex=0; localIndex<lmp->atom->nlocal; localIndex++)
	{
		localPositions[3*localIndex]=x[localIndex][0];
		localPositions[3*localIndex+1]=x[localIndex][1];
		localPositions[3*localIndex+2]=x[localIndex][2];

		localVelocities[3*localIndex]  =v[localIndex][0];
		localVelocities[3*localIndex+1]=v[localIndex][1];
		localVelocities[3*localIndex+2]=v[localIndex][2];

		localTags[localIndex]=lmp->atom->tag[localIndex];

		localRadii[localIndex] = r[localIndex];
	}

////////for(int localIndex=0; localIndex<lmp->atom->nlocal; localIndex++)
////////{
////////	std::cout<<localIndex<<"\t"<<localTags[localIndex]<<"\n";
////////}

	int nPoints[singleton::mpi().getSize()];
	int displacements[singleton::mpi().getSize()];

	int nTags[singleton::mpi().getSize()];
	int displacementsTags[singleton::mpi().getSize()];
////////	localNPoints=localNPoints*3;
////////}
        MPI_Allgather(&nLocalAtomsVec,1,MPI_INT,nPoints,1,MPI_INT,MPI_COMM_WORLD);	
        MPI_Allgather(&nLocalAtoms,1,MPI_INT,nTags,1,MPI_INT,MPI_COMM_WORLD);	
////////for(int rank=0;rank<singleton::mpi().getSize();rank++)
////////{
////////	std::cout<<nPoints[rank]<<"\t";	
////////}
////////std::cout<<"\n";
////////for(int rank=0;rank<singleton::mpi().getSize();rank++)
////////{
////////	std::cout<<nTags[rank]<<"\t";	
////////}
////////std::cout<<"\n";
	
	//double * globalPositions = new double [nAtoms*3];
	//double * globalVelocities = new double [nAtoms*3];
	int * globalTags = new int [nAtoms];

        int total_size=0;
        int totalSizeTags=0;
        for(int rank=0;rank<singleton::mpi().getSize();rank++)
        {
        	//std::cout<<i<<"\t"<<nPoints[i]<<"\n";;
        	displacements[rank]=total_size;
        	displacementsTags[rank]=totalSizeTags;

        	total_size=total_size+nPoints[rank];
        	totalSizeTags=totalSizeTags+nTags[rank];
        	//sendnum[i]=all_sizes[i];
        }
////////surface_points_global = new double[total_size];
        MPI_Allgatherv(localPositions,nLocalAtomsVec, MPI_DOUBLE, globalPositions,nPoints,displacements,MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localVelocities,nLocalAtomsVec, MPI_DOUBLE, globalVelocities,nPoints,displacements,MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localTags,nLocalAtoms, MPI_INT, globalTags,nTags,displacementsTags,MPI_INT, MPI_COMM_WORLD);
        MPI_Allgatherv(localRadii,nLocalAtoms, MPI_DOUBLE, globalRadii,nTags,displacementsTags,MPI_DOUBLE, MPI_COMM_WORLD);

	//int * reverseTags=new int[nAtoms+1];

        for(int i=0;i<nAtoms;i++)
        {
        	//std::cout<<globalTags[i]<<"\t"<<globalPositions[3*i]<<"\t"<<globalPositions[3*i+1]<<"\t"<<globalPositions[3*i+2]<<"\t"<<singleton::mpi().getRank()<<"\n";
		reverseTags[globalTags[i]]=i;
        }
}

int main(int argc, char* argv[]) {
	initialize(&argc, &argv);
	singleton::directories().setOutputDir( "./OLB_LAMMPS/" );

	OstreamManager clout(std::cout, "main");

	CLIreader args(argc, argv);

	LAMMPS_NS::LAMMPS *lmp;

	// custom argument vector for LAMMPS library
	const char *lmpargv[5];
	lmpargv[0] = "liblammps";//, "-log", "none"}; 
	lmpargv[1] = "-log";//, "-log", "none"}; 
	lmpargv[2] = "none";//, "-log", "none"}; 
	lmpargv[3] = "-screen";//, "-log", "none"}; 
	lmpargv[4] = "out.log";//, "-log", "none"}; 
	int lmpargc = sizeof(lmpargv)/sizeof(const char *);

	lmp = new LAMMPS_NS::LAMMPS(lmpargc, (char **)lmpargv, MPI_COMM_WORLD);
	// output numerical version string
	clout << "LAMMPS version ID: " << lmp->num_ver << std::endl;
	const int N = args.getValueOrFallback("--resolution", 4);

	const UnitConverterFromResolutionAndLatticeVelocity<T,DESCRIPTOR> converter(
			N,                      // resolution: number of voxels per charPhysL
			(T)   0.05,             // lattice velocity
			(T)   0.05,                // charPhysLength: reference length of simulation geometry
			(T)   2.25,                // charPhysVelocity: maximal/highest expected velocity during simulation in __m / s__
			(T)   1e-2,             // physViscosity: physical kinematic viscosity in __m^2 / s__
			(T)   1000              // physDensity: physical density in __kg / m^3__
			);
	converter.print();

	/* SETTING LAMMPS UP 
	 */
	lmp->input->one("units si"); 
	lmp->input->one("atom_style sphere"); 
	lmp->input->one("atom_modify map yes");

	auto domainI = makeDomainI();
	auto min = domainI->getMin();
	auto max = domainI->getMax();

	std::string regionCommand;
	regionCommand = "region BOX block "+std::to_string(min[0])+" "+std::to_string(max[0])+" "+std::to_string(min[1])+" "+std::to_string(max[1])+" "+std::to_string(min[2])+" "+std::to_string(max[2]);
	lmp->input->one(regionCommand);
	regionCommand = "region topBOX block "+std::to_string(min[0]+0.2)+" "+std::to_string(max[0]-0.2)+" "+std::to_string(min[1]+0.2)+" "+std::to_string(max[1]-0.2)+" "+std::to_string(1.*max[2]/3.)+" "+std::to_string(max[2]-0.2);
	lmp->input->one(regionCommand);
	lmp->input->one("create_box 1 BOX");

	lmp->input->one("change_box all boundary f f f");
	clout<<"created region in LAMMPS"<<std::endl;
	//return 0;
	std::shared_ptr<IndicatorF3D<T>> paddedDomainI(new IndicatorLayer3D<T>(*domainI, converter.getPhysDeltaX()));

	CuboidDecomposition<T,3> cuboidDecomposition(*paddedDomainI, converter.getPhysDeltaX(), singleton::mpi().getSize());
	cuboidDecomposition.setPeriodicity({true,true,false});  // Periodic boundary condition in x and y directions 
	BlockLoadBalancer<T> loadBalancer(cuboidDecomposition);

	SuperGeometry<T,3> sGeometry(cuboidDecomposition, loadBalancer);
	prepareGeometry(converter, sGeometry);

	SuperLattice<T,DESCRIPTOR> sLattice(sGeometry);
	prepareLattice(sLattice, converter, sGeometry);

	SuperPorousElementEmbeddingO fsiEmbeddingO(sLattice);
	fsiEmbeddingO.registerElementType<SpherePorosityF>(sGeometry.getMaterialIndicator(1));


	lmp->input->one("lattice sc 0.05");
	//lmp->input->one("create_atoms 1 mesh makeASCII/ellipsoid.stl units box");
	lmp->input->one("create_atoms 1 region topBOX");
	lmp->input->one("set type 1 diameter 0.05");
	lmp->input->one("set type 1 density 9000");

	lmp->input->one("dump total_force all custom 1 check.dat id mass x y z radius vx vy vz");
	lmp->input->one("run 0");
	lmp->input->one("undump total_force");
	//return 0;
	//return 0;
	lmp->input->one("pair_style granular");
	lmp->input->one("pair_coeff * * hooke 15707.963267948964 0.69 tangential linear_history 10471.975511965977 1.0 0.5 rolling sds 200.0 100.0 0.1 twisting marshall damping coeff_restitution");
	
	lmp->input->one("comm_modify vel yes");
	lmp->input->one("fix 2 all nve/sphere");
	lmp->input->one("fix grav all gravity 9.8 vector 0 0 -1");
	lmp->input->one("timestep 0.0001");
	std::string wallCommand;
	wallCommand = "fix side_wall_zh all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 zplane "+std::to_string(max[2])+" NULL contacts";
	lmp->input->one(wallCommand);
	wallCommand = "fix side_wall_zl all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 zplane "+std::to_string(min[2])+" NULL contacts";
	lmp->input->one(wallCommand);
        wallCommand = "fix side_wall_yh all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 yplane "+std::to_string(max[1])+" NULL contacts";
        lmp->input->one(wallCommand);
        wallCommand = "fix side_wall_yl all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 yplane "+std::to_string(min[1])+" NULL contacts";
        lmp->input->one(wallCommand);
        wallCommand = "fix side_wall_xh all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 xplane "+std::to_string(max[0])+" NULL contacts";
        lmp->input->one(wallCommand);
        wallCommand = "fix side_wall_xl all wall/gran granular hooke 55707.963267948964 0.9 tangential linear_history 1e-6 1.0 0.0 xplane "+std::to_string(min[0])+" NULL contacts";
        lmp->input->one(wallCommand);

	lmp->input->one("dump configs all custom 500 config*.dat id diameter mass x y z vx vy vz radius");
	lmp->input->one("thermo 100");

	lmp->input->one("fix 5 all external pf/array 1"); // This is the LAMMPS fix which lets you add an external force to the system
	lmp->input->one("velocity all create 3000.0 4928459 rot yes dist gaussian");
	//lmp->input->one("velocity all set -0.5 0.5 0.2");
	lmp->input->one("run 0");
	lmp->input->one("neigh_modify check yes");
	lmp->input->one("run 10000");
	const unsigned nSpheres = lmp->atom->natoms;
	int nAtoms = lmp->atom->natoms;
	double * globalPositions = new double [nAtoms*3];
	double * globalVelocities = new double [nAtoms*3];
	double * globalRadii = new double [nAtoms];
	int * reverseTags=new int[nAtoms+1];
	Vector<T,3> spherePhysR[nSpheres] ;
	T sphereR[nSpheres] ;
	getGlobalPositionVelocity(lmp,globalPositions,globalVelocities,globalRadii,reverseTags);
        for (unsigned iSphere=0; iSphere < nSpheres; ++iSphere) {

		int indexInGlobalArray=reverseTags[iSphere+1];
		//clout<<iSphere<<"\t"<<indexInGlobalArray<<"\t"<<globalRadii[indexInGlobalArray]<<"\t"<<Vector<double,3> {globalPositions[indexInGlobalArray*3],globalPositions[indexInGlobalArray*3+1],globalPositions[indexInGlobalArray*3+2]}<<std::endl;
        	auto sphereParams = makeParametersD<T,descriptors::SPATIAL_DESCRIPTOR<3>>(
        			fields::fsi::ELEMENT_TAG{},   1+iSphere,
        			fields::fsi::ELEMENT_PIVOT{}, Vector<double,3> {globalPositions[indexInGlobalArray*3],globalPositions[indexInGlobalArray*3+1],globalPositions[indexInGlobalArray*3+2]},
        			SpherePorosityF::RADIUS{}, globalRadii[indexInGlobalArray]
        			);
        	fsiEmbeddingO.add(sphereParams);
        }
	//return 0;
	SuperPorousElementReductionO<T,DESCRIPTOR,fields::fsi::ELEMENT_FORCE> fsiReductionO(
			sLattice,
			sGeometry.getMaterialIndicator(1));
	fsiReductionO.resize(nSpheres);
	fsiReductionO.addCollectionO(meta::id<CollectPorousBoundaryForceO>{});

	sLattice.setParameter<fields::converter::PHYS_LENGTH>(
			1/converter.getConversionFactorLength());
	sLattice.setParameter<fields::converter::PHYS_VELOCITY>(
			1/converter.getConversionFactorVelocity());
	sLattice.setParameter<fields::converter::PHYS_DELTA_X>(
			converter.getPhysDeltaX());

	sLattice.setProcessingContext(ProcessingContext::Simulation);
	fsiEmbeddingO.initialize();

	util::Timer<T> timer(converter.getLatticeTime(maxPhysT),
			sGeometry.getStatistics().getNvoxel());
	timer.start();
	auto fix = lmp->modify->get_fix_by_id("5");
	double **fexternal = nullptr;
	//int nAtoms = lmp->atom->natoms;
	for (std::size_t iT = 0; iT <= converter.getLatticeTime(maxPhysT); ++iT) {

		setBoundaryValues( sLattice, converter, iT, sGeometry );
		sLattice.collideAndStream();

		//clout << "Time taken by Collid_and_stream: "<< duration.count() << " microseconds" << std::endl;

		getResults(sLattice, converter, sGeometry, timer, iT);


		fsiReductionO.apply();
		lmp->input->one("run 1");

		int tmp;
		//start = std::chrono::high_resolution_clock::now();
		fexternal = (double **)fix->extract("fexternal",tmp);
		//clout << "Time taken by run: "<< duration.count() << " microseconds" << std::endl;

		
		getGlobalPositionVelocity(lmp,globalPositions,globalVelocities,globalRadii,reverseTags);
		//clout<<fsiReductionO.getElementCount()<<std::endl;
	  	if (fsiReductionO.rankDoesFSI()) 
	  	{
			for (unsigned iElement=1; iElement <= fsiReductionO.getElementCount(); ++iElement) 
			{
				auto iSphere = fsiEmbeddingO.getField<fields::fsi::ELEMENT_TAG>(iElement-1)[0];
				auto f = converter.getPhysForce(fsiReductionO.getField<fields::fsi::ELEMENT_FORCE>(iElement));
				int indexInGlobalArray=reverseTags[iSphere];
				auto position=Vector{globalPositions[3*indexInGlobalArray],globalPositions[3*indexInGlobalArray+1],globalPositions[3*indexInGlobalArray+2]};
				auto velocity=Vector{globalVelocities[3*indexInGlobalArray],globalVelocities[3*indexInGlobalArray+1],globalVelocities[3*indexInGlobalArray+2]};
				if(lmp->atom->map(iSphere)>-1)
				{
					fexternal[lmp->atom->map(iSphere)][0]=f[0];
					fexternal[lmp->atom->map(iSphere)][1]=f[1];
					fexternal[lmp->atom->map(iSphere)][2]=f[2];
				}
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_PIVOT>(iElement-1, position);
				fsiEmbeddingO.setField<fields::fsi::ELEMENT_U_TRANSLATION>(iElement-1, velocity);
			}
	  		fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_PIVOT>>(ProcessingContext::Simulation);
	  		fsiEmbeddingO.setProcessingContext<Array<fields::fsi::ELEMENT_U_TRANSLATION>>(ProcessingContext::Simulation);
	  	}
		//clout << "Time taken by fapply: "<< duration.count() << " microseconds" << std::endl;

		fsiEmbeddingO.apply();
	}

	sLattice.setProcessingContext(ProcessingContext::Evaluation);

	timer.stop();
	timer.printSummary();
	delete lmp;

	// stop MPI environment
	MPI_Finalize();
	return 0;
}
