function has_DataType_or_UnionAll(obj, name, pc = ())
    for pn in propertynames(obj)
        prop = getproperty(obj, pn)
        pc_full = (pc..., ".", pn)
        pc_string = name*string(join(pc_full))
        if prop isa DataType
            @warn "$pc_string::$(typeof(prop)) is a DataType"
            return true
        elseif prop isa UnionAll
            @warn "$pc_string::$(typeof(prop)) is a UnionAll"
            return true
        else
            has_DataType_or_UnionAll(prop, name, pc_full)
        end
    end
    return false
end
macro has_DataType_or_UnionAll(obj)
    return :(has_DataType_or_UnionAll($(esc(obj)), $(string(obj))))
end
