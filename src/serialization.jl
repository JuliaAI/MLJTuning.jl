MLJModelInterface.save(::MLJTuning.EitherTunedModel, fitresult::Machine) =
    serializable(fitresult)

function MLJModelInterface.restore(::MLJTuning.EitherTunedModel, fitresult)
    fitresult.fitresult = restore(fitresult.model, fitresult.fitresult)
    return fitresult
end