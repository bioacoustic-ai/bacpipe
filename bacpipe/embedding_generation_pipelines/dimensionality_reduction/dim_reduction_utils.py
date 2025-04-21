def comppute_reduction(paths, orig_embeddings, name, reducer, **kwargs):
    """
    Compute the dimensionality reduction for the embeddings.

    Parameters
    ----------
    orig_embeddings : dict
        dictionary containing all the embeddings for each model and labels
    name : string
        name of the dimensionality reduction method
    reducer : class
        class of the dimensionality reduction method
    **kwargs : dict
        dictionary containing the configuration of the dimensionality reduction method

    Returns
    -------
    processed_embeds : dict
        dictionary containing the processed embeddings
    """
    processed_embeds = {}
    for model, embed in tqdm(
        orig_embeddings.items(),
        desc=f"calculating dimensionality reduction for {name}",
        position=0,
        leave=False,
    ):
        file = folder.joinpath(f"{name}_{model}.npy")
        if not file.exists():
            processed_embeds[model] = {}
            processed_embeds[model]["all"] = reducer.fit_transform(embed["all"])

            processed_embeds[model]["labels"] = embed["labels"]
            if "label_dict" in embed.keys():
                processed_embeds[model]["label_dict"] = embed["label_dict"]
            np.save(file, processed_embeds[model])
        else:
            processed_embeds[model] = np.load(
                file,
                allow_pickle=True,
            ).item()
    return processed_embeds


def define_2d_reducer(reducer_2d_conf, verbose=True):
    if reducer_2d_conf["name"] == "2dumap":
        reducer = umap.UMAP(**list(reducer_2d_conf.values())[-1], verbose=verbose)
    else:
        assert False, "Reducer not implemented"
    return reducer


def get_reduced_embeddings_by_label(embed, model, reduc_embeds, label_file=None):
    # if label_file:
    reduc_embeds[model]["split"] = {
        k: reduc_embeds[model]["all"][embed["labels"] == v]
        for k, v in embed["label_dict"].items()
    }
    # else:
    #     reduc_embeds[model]["split"] = {  # TODO test if this works
    #         k: np.split(
    #             reduc_embeds[model]["all"], list(embed["label_dict"].values())[1:]
    #         )
    #         for k in embed["label_dict"].keys()
    #     }
    return reduc_embeds


def reduce_to_2d(embeds, name, reducer_2d_conf=None, label_file=None, **kwargs):
    if not np_embeds_path.joinpath(f"{name}.npy").exists():
        reduc_embeds = {}
        for model, embed in tqdm(
            embeds.items(),
            desc="calculating dimensionality reduction",
            position=0,
            leave=False,
        ):
            reduc_embeds[model] = {}
            reducer = define_2d_reducer(reducer_2d_conf)

            reduc_embeds[model]["all"] = reducer.fit_transform(embed["all"])

            reduc_embeds = get_reduced_embeddings_by_label(
                embed, model, reduc_embeds, label_file
            )

            reduc_embeds[model]["labels"] = embed["labels"]

        np.save(np_embeds_path.joinpath(f"{name}.npy"), reduc_embeds)
    else:
        reduc_embeds = np.load(
            np_embeds_path.joinpath(f"{name}.npy"), allow_pickle=True
        ).item()
    return reduc_embeds
