import jax
import jax.numpy as jnp

# Please note: this implementation is a translation of the RRMPG implementation
# of GR4J. The original implementation can be found here:
# https://github.com/kratzert/RRMPG
# Hopefully we can we can look to integrating jax models with RRMPG in the future

#################################################################################################
####################################################################################################

@jax.jit
def run_gr4j_jax(forcing,params_dict):
    '''
    Implementation of the GR4J model in JAX
    Inputs:
        - forcing: (n_timesteps,2) a jax array with values for rain at each timestep in the first column and evaporation (at each timestep) in the second column. e.g., you could create this from numpy arrays `rain` and `et` using: `inputs = jnp.stack([rain,et],axis=1)`.
        - params_dict: an dictionary with the parameters of the model:
            - s_init: s storage init
            - r_init: r storage init
            - X1
            - X2
            - X3
            - X4: max value of 9
    Outputs:
        - qsim: (n_timesteps,) a jax array with simulated discharge at each timestep
        - s: (n_timesteps,) a jax array with the s storage at each timestep
        - r: (n_timesteps,) a jax array with the r storage at each timestep
    '''

    # Some default parameters taken by eyeballing Buzacott et al. (2019)
    X1 = params_dict.get('X1',800)
    X2 = params_dict.get('X2',-0.75)
    X3 = params_dict.get('X3',180)
    X4 = params_dict.get('X4',1.3)

    # # compute the unit hydrographs
    num_uh1 = jnp.int32(jnp.ceil(X4))
    num_uh2 = jnp.int32(jnp.ceil(2*X4 + 1))

    # create ords for a resonable number of unit hydrograph
    # we will allow for a max value of 10 for X4
    # I think the only way with jax is to set a limit
    # its not ideal and probably not the most efficient but it works
    uh1_range = jnp.arange(11) - (11 - (num_uh1 + 1))
    uh2_range = jnp.arange(21) - (21 - (num_uh2 + 1))

    # calculate ordinates of unit hydrographs - s curves first
    calc_sc1_map = jax.vmap(calc_s_curve_1, in_axes=(0, None))
    calc_sc2_map = jax.vmap(calc_s_curve_2, in_axes=(0, None))
    s_curves_1 = calc_sc1_map(uh1_range, X4)
    s_curves_2 = calc_sc2_map(uh2_range, X4)

    uh1_ord = s_curves_1[1:] - s_curves_1[:-1]
    uh2_ord = s_curves_2[1:] - s_curves_2[:-1]

    # Unpack the parameters and define the carry
    carry = {
        's_store': params_dict.get('s_init',0.5) * X1,
        'r_store': params_dict.get('r_init',0.5) * X3,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'uh1_ords': uh1_ord,
        'uh2_ords': uh2_ord,
        'uh1': jnp.zeros((uh1_ord.size,)),
        'uh2': jnp.zeros((uh2_ord.size,)),
        'num_uh1': num_uh1,
        'num_uh2': num_uh2,
    }

    # to achieve the stepping through time in jax, we will use lax.scan
    carry_out, outputs = jax.lax.scan(gr4j_time_update, carry, forcing)

    return outputs[:,0], outputs[:,1], outputs[:,2]

#################################################################################################
####################################################################################################

@jax.jit
def gr4j_time_update(carry, t_input):
    '''
    This function implements the time stepping component of the GR4J model.
    Inputs:
        - carry: a dictionary with parameters and internal model states that 
        need to be carried over to the next timestep
            - s_store: s storage
            - r_store: r storage
            - X1
            - X2
            - X3
            - X4: max value of 9
            - uh1_ords: ordinates of the first unit hydrograph
            - uh2_ords: ordinates of the second unit hydrograph
            - uh1: values of the first unit hydrograph
            - uh2: values of the second unit hydrograph
            - num_uh1: number of ordinates in the first unit hydrograph
            - num_uh2: number of ordinates in the second unit hydrograph
        - t_input: (2,) a jax array with values for rain at this timestep 
        in the first column and evaporation in the second column.
    
    You will note that variables and comments are carried over from the RRMPG 
    implmentation which was used as the base of this function. We thank the 
    authors of RRMPG and hope that the jax translation can be helpful for others.
    '''
    # first calculate the net precipitation effect on stores
    is_gain = t_input[0] >= t_input[1]
    p_n_gain = t_input[0] - t_input[1]
    p_n_loss = jnp.float32(0.0)
    pe_n_loss = t_input[1] - t_input[0]

    # calculate the evaporation effect on production store
    e_s_gain = jnp.float32(0.0)
    e_s_loss = jnp.divide(
        carry['s_store'] * (2 - carry['s_store']/carry['X1']) * jnp.tanh(pe_n_loss / carry['X1']),
        1 + (1 - carry['s_store'] / carry['X1']) * jnp.tanh(pe_n_loss / carry['X1'])
    ) 

    # calculate the precipitation effect on production store
    p_s_gain = jnp.divide(
        carry['X1'] * (1 - (carry['s_store']  / carry['X1'])**2) * jnp.tanh(p_n_gain/carry['X1']),
        1 + (carry['s_store']  / carry['X1']) * jnp.tanh(p_n_gain / carry['X1'])
    ) 
    p_s_loss = jnp.float32(0.0) 

    # avoiding if statements
    p_n = p_n_gain * is_gain + p_n_loss * (1 - is_gain)
    p_s = p_s_gain * is_gain + p_s_loss * (1 - is_gain)
    e_s = e_s_gain * is_gain + e_s_loss * (1 - is_gain)

    # update the s store
    tmp_s_store = carry['s_store'] + p_s - e_s

    # calculate percolation from actual storage level
    perc = tmp_s_store * (1 - (1 + (4/9 * tmp_s_store / carry['X1'])**4)**(-0.25))

    # final update of the production store for this timestep
    carry['s_store'] = tmp_s_store - perc
    
    # total quantity of water that reaches the routing
    p_r = perc + (p_n - p_s)
    
    # split this water quantity by .9/.1 for diff. routing (UH1 & UH2)
    p_r_uh1 = 0.9 * p_r 
    p_r_uh2 = 0.1 * p_r
    
    # update state of rain, distributed through the unit hydrographs
    tmp_uh1_1 = carry['uh1'][1:] + carry['uh1_ords'][:-1] * p_r_uh1
    tmp_uh1_2 = jnp.array(carry['uh1_ords'][-1] * p_r_uh1)
    carry['uh1'] = jnp.hstack([tmp_uh1_1,tmp_uh1_2])
    
    tmp_uh2_1 = carry['uh2'][1:] + carry['uh2_ords'][:-1] * p_r_uh2
    tmp_uh2_2 = jnp.array(carry['uh2_ords'][-1] * p_r_uh2)
    carry['uh2'] = jnp.hstack([tmp_uh2_1,tmp_uh2_2])

    # calculate the groundwater exchange F (eq. 18)
    gw_exchange = carry['X2'] * (carry['r_store'] / carry['X3']) ** 3.5
    
    # update routing store
    tmp_r_store = jnp.maximum(0, carry['r_store'] + carry['uh1'][11-(carry['num_uh1']+1)] + gw_exchange)

    # outflow of routing store
    q_r = tmp_r_store * (1 - (1 + (tmp_r_store / carry['X3'])**4)**(-0.25))
    
    # subtract outflow from routing store level
    carry['r_store'] = tmp_r_store - q_r
    
    # calculate flow component of unit hydrograph 2
    q_d = jnp.maximum(0, carry['uh2'][21-(carry['num_uh2']+1)] + gw_exchange)
    
    # total discharge of this timestep
    qsim = q_r + q_d

    return carry, jnp.stack([qsim, carry['s_store'], carry['r_store']], axis=0)
           
#################################################################################################
####################################################################################################

@jax.jit
def calc_s_curve_1(timestep, X4):
    # working around jax's dislike of if statements
    out = jnp.piecewise(
        jnp.float32(timestep), # output defaults to the type of this
        [
            timestep <= 0,
            (timestep < X4) & (timestep > 0)
        ], # conditions list
        [
            lambda x, x4: jnp.float32(0.0),
            lambda x, x4: (x / x4) ** 2.5,
            lambda x ,x4: jnp.float32(1.0)
        ], # function list + 1 for default
        x4 = X4
    )
    return out

@jax.jit
def calc_s_curve_2(timestep, X4):
    out = jnp.piecewise(
        jnp.float32(timestep), 
        [
            timestep <= 0,
            (timestep <= X4) & (timestep > 0),
            (timestep < 2 * X4) & (timestep > X4)
        ], # conditions list
        [
            lambda x, x4: jnp.float32(0.0),
            lambda x, x4: 0.5 * ((x / x4) ** 2.5),
            lambda x, x4: 1.0 - 0.5 * ((2.0 - (x / x4)) ** 2.5),
            lambda x, x4: jnp.float32(1.0)
        ], # function list + 1 for default
        x4 = X4
    )
    return out

#################################################################################################
####################################################################################################



