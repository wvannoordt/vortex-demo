#include <chrono>
#include <spade.h>
#include <PTL.h>

template <typename data_t, const std::size_t ar_size> using array = spade::ctrs::array<data_t, ar_size>;

int main(int argc, char** argv)
{
    //initialize MPI
    spade::parallel::mpi_t group(&argc, &argv);
    
    //Get the input file
    std::vector<std::string> args;
    for (auto i: range(0, argc)) args.push_back(std::string(argv[i]));
    if (args.size() < 2)
    {
        if (group.isroot()) print("Please provide an input file name!");
        return 1;
    }
    std::string input_filename = args[1];
    
    //read the input file
    PTL::PropertyTree input;
    input.Read(input_filename);
    
    const real_t targ_cfl         = input["Config"]["cfl"];
    const int    nt_max           = input["Config"]["nt_max"];
    const int    nt_skip          = input["Config"]["nt_skip"];
    const int    checkpoint_skip  = input["Config"]["ck_skip"];
    const int    nx               = input["Config"]["nx_cell"];
    const int    ny               = input["Config"]["ny_cell"];
    const int    nxb              = input["Config"]["nx_blck"];
    const int    nyb              = input["Config"]["ny_blck"];
    const int    nguard           = input["Config"]["nguard"];
    const real_t xmin             = input["Config"]["xmin"];
    const real_t xmax             = input["Config"]["xmax"];
    const real_t ymin             = input["Config"]["ymin"];
    const real_t ymax             = input["Config"]["ymax"];
    const bool   do_output        = input["Config"]["output"];
    const std::string init_file   = input["Config"]["init_file"];
    const real_t u0               = input["Fluid"]["u0"];
    const real_t deltau           = input["Fluid"]["deltau"];
    const real_t gamma            = input["Fluid"]["gamma"];
    const real_t b                = input["Fluid"]["b"];
    const real_t cp               = input["Fluid"]["cp"];
    const real_t theta_d          = input["Fluid"]["theta_d"];
    
    const real_t xc = 0.5*(xmin+xmax);
    const real_t yc = 0.5*(ymin+ymax);
    
    array<int, 2> num_blocks       (nxb, nyb);
    array<int, 2> cells_in_block   (nx, ny);
    array<int, 2> exchange_cells   (nguard, nguard);
    spade::bound_box_t<real_t, 2> bounds;
    bounds.min(0) =  xmin;
    bounds.max(0) =  xmax;
    bounds.min(1) =  ymin;
    bounds.max(1) =  ymax;
    
    
    //restart directory
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    //cartesian coordinate system
    spade::coords::identity<real_t> coords;
    
    //create grid
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    
    
    //create arrays residing on the grid
    prim_t fill1 = 0.0;
    spade::grid::grid_array prim (grid, fill1);
    
    flux_t fill2 = 0.0;
    spade::grid::grid_array rhs  (grid, fill2);
    
    //===============================================================
    //Note that everything above this point is more or less mandatory
    //===============================================================
    
    //define the initial condition
    const real_t sintheta = std::sin(theta_d*spade::consts::pi/180.0);
    const real_t costheta = std::cos(theta_d*spade::consts::pi/180.0);
    const real_t u_theta  = u0*costheta;
    const real_t v_theta  = u0*sintheta;
    auto ini = [&](const spade::ctrs::array<real_t, 3> x) -> prim_t
    {
        prim_t output;
        const real_t r      = std::sqrt((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc));
        const real_t upmax  = deltau*u0;
        const real_t expfac = std::exp(0.5*(1.0-((r*r)/(b*b))));
        const real_t ur     = (1.0/b)*deltau*u0*r*expfac;
        const real_t rhor   = std::pow(1.0 - 0.5*(air.gamma-1.0)*deltau*u0*deltau*u0*expfac, 1.0/(air.gamma - 1.0));
        const real_t pr     = std::pow(rhor, air.gamma)/air.gamma;
        const real_t theta_loc = std::atan2(x[1], x[0]);
        output.p() = pr;
        output.T() = pr/(rhor*air.R);
        output.u() = u_theta - ur*std::sin(theta_loc);
        output.v() = v_theta + ur*std::cos(theta_loc);
        output.w() = 0.0;
        return output;
    };
    
    //fill the initial condition
    spade::algs::fill_array(prim, ini);
    
    //fill the guards
    grid.exchange_array(prim);
    
    //if a restart file is specified, read the data, fill the array, and fill guards
    if (init_file != "none")
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_file, prim);
        if (group.isroot()) print("Init done.");
        grid.exchange_array(prim);
    }
    
    //using the 2nd-order centered KEEP scheme
    spade::convective::totani_lr        tscheme(air);
    
    //define an element-wise kernel that returns the acoustic wavespeed for CFL calculation
    struct get_u_t
    {
        const spade::fluid_state::ideal_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        get_u_t(const spade::fluid_state::ideal_gas_t<real_t>& gas_in) {gas = &gas_in;}
        real_t operator () (const prim_t& q) const
        {
            return sqrt(gas->gamma*gas->R*q.T()) + sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w());
        }
    } get_u(air);
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    spade::reduce_ops::reduce_max<real_t> sum_op;
    
    
    
    //calculate timestep
    real_t time0 = 0.0;
    const real_t dx = spade::utils::min(grid.get_dx(0), grid.get_dx(1), grid.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;
    
    //define the conservative variable transformation
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air);
    
    //define the residual calculation
    auto calc_rhs = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        grid.exchange_array(q);
        
        //compute the convective terms
        spade::pde_algs::flux_div(q, rhs, tscheme);
    };
    
    //define the time integrator
    spade::time_integration::rk2 time_int(prim, rhs, time0, dt, calc_rhs, trans);
    
    spade::utils::mtimer_t tmr("advance");
    
    //time loop
    for (auto nt: range(0, nt_max+1))
    {
        //cacluate the maximum wavespeed |u|+a
        const real_t umax   = spade::algs::transform_reduce(prim, get_u, max_op);        
        
        //print some nice things to the screen
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print(
                "nt: ",  spade::utils::pad_str(nt, pn),
                "cfl:",  spade::utils::pad_str(cfl, pn),
                "umax:", spade::utils::pad_str(umax, pn),
                "dx: ",  spade::utils::pad_str(dx, pn),
                "dt: ",  spade::utils::pad_str(dt, pn)
            );
        }
        
        //output the solution
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "prims"+nstr;
            if (do_output) spade::io::output_vtk("output", filename, prim);
            if (group.isroot()) print("Done.");
        }
        
        //output a restart file if needed
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = "checkpoint/"+filename+".bin";
            if (do_output) spade::io::binary_write(filename, prim);
            if (group.isroot()) print("Done.");
        }
        
        //advance the solution
        tmr.start("advance");
        time_int.advance();
        tmr.stop("advance");
        
        if (group.isroot()) print(tmr);
        
        //check for solution divergence
        if (std::isnan(umax))
        {
            if (group.isroot())
            {
                print("A tragedy has occurred!");
            }
            group.sync();
            return 155;
        }
    }
    
    return 0;
}
