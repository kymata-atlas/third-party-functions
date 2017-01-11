function out = Spectrum2ExcitationPattern025( f, in )

InputLevels = InputLevelPerERB(f,in);

ExcitationLevels = OutputPowerAtERBNumbers025(InputLevels, f, in);

out = ExcitationLevels;