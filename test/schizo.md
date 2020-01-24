@tlienart I have a general trick to remove dependencies in any interface, when these dependencies are needed only for *functions* (not types). In our case this suffices, because the only types that need to be referenced in MLJModelInterface are the ScientificTypes, which we can move into a ScientificTypesTypesOnlyPackage which will give it trivial overhead. 

Here's the general trick:  First, define a mutable flag `INTERFACE_FUNCTIONALITY` in the lightweight interface package - let's call it Light.jl - which is for declaring whether we are in `Dummy()` or `Live()` mode, and to set it there to  `Dummy()`:

**LightInterface.jl** package

```julia
abstract type Mode end
struct Dummy <: Mode end
struct Live <: Mode end
const  INTERFACE_FUNCTIONALITY = Ref{Mode}(Dummy())
```

Any function we want from some dependent package - `matrix` from Tables.jl`, say - we define right there in LightInterface.jl  but dispatch it on the flag value, giving it a dummy return value when the flag is `Dummy()`. 

**Client* package

```julia
matrix(t) = matrix(t, INTERFACE_FUNCTIONALITY[])# 
matrix(t, ::Dummy) = error("Only `LightInterface` loaded. Do `import Interface`.")
```

In any client can now import LightInterface and make calls like `matrix(t)` (and `t` could be anything). Of course executing the client's code will fail at this point. However, we fix this in the full interface package, let's call it Interface.jl, by completing the definition of `matrix` in the live case, now that Tables.jl is available, and in the `__init__` function of `Interface.jl` the flag has been set to `Live()`:

**Interface.jl* package

```julia
function __init__()
    LightInterface.INTERFACE_FUNCTIONALITY[] = LightInterface.Live()
end

using ..LightInterface
import Tables  # the elephant LightInterface didn't want

LightInterface.matrix(t, ::Live) = Tables.matrix(t)
```

Now, if the client package *and* Interface.jl are both loaded, it can actually be used.

What do you think?
 
Here's a proof of concept:

---

### THE LIGHT INTERFACE

```
module LightInterface

export Dummy, Live
export matrix

abstract type Mode end
struct Dummy <: Mode end
struct Live <: Mode end
const  INTERFACE_FUNCTIONALITY = Ref{Mode}(Dummy())

matrix(t) = matrix(t, INTERFACE_FUNCTIONALITY[])# 
matrix(t, ::Dummy) =
    error("Only `LightInterface` loaded. Do `import Interface`.")

end
```

### MODULE IMPLEMENTING THE LIGHT INTERFACE

```julia
module Client

export double

using ..LightInterface

double(t) = 2*matrix(t)

end
```
### CAN LOAD CLIENT CODE; JUST CANT USE IT

```julia
julia> using .Client
julia> t = (x1=rand(3), x2=rand(3)) # a table
julia> double(t)
ERROR: Only `LightInterface` loaded. Do `import Interface`.
``
### THE FULL INTERFACE

```julia
module Interface

export matrix

using ..LightInterface
import Tables  # the elephant LightInterface didn't want

# In a package this would be in the __init__ function:
LightInterface.INTERFACE_FUNCTIONALITY[] = LightInterface.Live()

LightInterface.matrix(t, ::Live) = Tables.matrix(t)

end
```
### IMPORTING THE FULL INTERFACE ENABLES FUNCTIONALITY:

```julia
julia> using .Interface
julia> double(t)
3×2 Array{Float64,2}:
 0.195865  0.0202382
 1.30479   0.329098 
 0.150455  1.7458  
```
