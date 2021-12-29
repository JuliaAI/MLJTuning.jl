@recipe function f(mach::MLJBase.Machine{<:EitherTunedModel})
    rep = report(mach)
    measurement = string(typeof(rep.best_history_entry.measure[1]))
    r = rep.plotting
    z = r.measurements
    X = r.parameter_values
    guides = r.parameter_names
    scales = r.parameter_scales
    n = size(X, 2)
    indices = LinearIndices((n, n))'

    layout := (n, n)
    size --> (275*n, 250*n)
    label := ""

    framestyle := :none
    for i in 1:n-1, j in i+1:n
        @series begin
            subplot := indices[i, j]
            ()
        end
    end

    framestyle := :box
    seriestype := :scatter
    for i in 1:n
        x = X[:, i]
        xsc = scales[i]
        xsc == :none && (x = string.(x))
        xscale := (xsc in (:custom, :linear, :none) ? :identity : xsc)

        @series begin
            subplot := indices[i, i]
            xguide := (i == n) ? guides[i] : ""
            yguide := measurement
            x, z
        end

        for j in 1:i-1
            y = X[:, j]
            ysc = scales[j]
            ysc == :none && (y = string.(y))
            ms = get(plotattributes, :markersize, 4)

            @series begin
                subplot := indices[i, j]
                yscale := (ysc in (:custom, :linear, :none) ? :identity : ysc)
                xguide := (i == n) ? guides[j] : ""
                yguide := (j == 1) ? guides[i] : ""
                markersize := _getmarkersize(ms, z)
                marker_z := z
                colorbar := false
                y, x
            end
        end
    end
end

function _getmarkersize(ms, z)
    ret = sqrt.(abs.(z))
    minz, maxz = extrema(x for x in ret if isfinite(x))
    ret .-= minz
    ret ./= maxz - minz
    4ms .* ret .+ 1
end
