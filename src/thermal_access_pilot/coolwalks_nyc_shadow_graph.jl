using JSON
using Dates
using TimeZones
using DataFrames
using GeoDataFrames
using Extents
using ArchGDAL
using Graphs
using MetaGraphs
using CoolWalksUtils
using ShadowGraphs
using CompositeBuildings
using TreeLoaders
using MinistryOfCoolWalks

function load_lidar_trees(path, extent)
    df = GeoDataFrames.read(path)
    rename!(df, Dict(:tree_id => :id, :height_m => :height, :radius_m => :radius))
    CoolWalksUtils.apply_extent!(df, extent)
    CoolWalksUtils.set_observatory!(df, "NYCLidarTreesObservatory", tz"America/New_York"; source=[:geometry])
    TreeLoaders.check_tree_dataframe_integrity(df)
    return df
end

function load_nyc_buildings(path, extent)
    df = GeoDataFrames.read(path)
    rename!(df, Dict(:doitt_id => :id, :heightroof => :height))
    df.height = parse.(Float64, string.(df.height))
    filter!(:height => >(0.0), df)
    transform!(df, [:geometry, :id] => ByRow(CompositeBuildings.split_multi_poly) => [:geometry, :id])
    df = flatten(df, [:geometry, :id])
    transform!(df, :height => ByRow(h -> h * 0.3048) => :height)
    CoolWalksUtils.apply_extent!(df, extent; source=[:geometry])
    CoolWalksUtils.set_observatory!(df, "NewYorkBuildingsObservatory", tz"America/New_York"; source=[:geometry])
    CompositeBuildings.check_building_dataframe_integrity(df)
    return df
end

function main(config_path::String)
    cfg = JSON.parsefile(config_path)
    extent = Extent(X=(Float64(cfg["min_lon"]), Float64(cfg["max_lon"])), Y=(Float64(cfg["min_lat"]), Float64(cfg["max_lat"])))

    buildings = load_nyc_buildings(String(cfg["buildings_path"]), extent)
    trees = load_lidar_trees(String(cfg["trees_path"]), extent)

    local_dt = DateTime(String(cfg["timestamp_local"]))
    building_shadows = CompositeBuildings.cast_shadows(buildings, local_dt)
    tree_shadows = TreeLoaders.cast_shadows(trees, local_dt)
    all_shadows = vcat(
        select(building_shadows, [:id, :geometry]),
        select(tree_shadows, [:id, :geometry]),
    )

    g = shadow_graph_from_download(
        :bbox;
        minlat=Float64(cfg["min_lat"]),
        minlon=Float64(cfg["min_lon"]),
        maxlat=Float64(cfg["max_lat"]),
        maxlon=Float64(cfg["max_lon"]),
        network_type=:walk,
        timezone=tz"America/New_York",
    )

    add_shadow_intervals!(g, all_shadows; clear_old_shadows=true)

    for v in vertices(g)
        set_prop!(g, v, :vertex_id, Int(v))
    end
    for e in edges(g)
        if !has_prop(g, e, :sg_shadow_length)
            set_prop!(g, e, :sg_shadow_length, 0.0)
        end
    end

    export_shadow_graph_to_csv(
        String(cfg["output_prefix"]),
        g;
        edge_props=All(),
        vertex_props=[:vertex_id, :sg_lon, :sg_lat],
        graph_props=[:sg_observatory],
    )
end

main(ARGS[1])
