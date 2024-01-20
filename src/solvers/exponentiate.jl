function exponentiate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  region_updates,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  default_updater_kwargs = (;
    krylovdim=30,  #from here only solver kwargs
    maxiter=100,
    verbosity=0,
    tol=1E-12,
    ishermitian=true,
    issymmetric=true,
    eager=true,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedence
  result, exp_info = exponentiate(
    projected_operator![],
    time_step,
    init;
    updater_kwargs...
  )
  return result, (; info=exp_info)
end
