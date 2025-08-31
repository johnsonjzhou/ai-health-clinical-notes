import pandas as pd

class CONCEPT_GROUP:
    """
    SNOMED CT concept group codes
    Get more from: https://ontoserver.csiro.au/site/our-solutions/shrimp/
    """
    DISEASE: str = "64572001"
    PRODUCT_NAME: str = "774167006"


def read_tsv(file_path: str) -> pd.DataFrame:
    """
    This function data from a Tab Separated Values (TSV) file and converts
    it into a Pandas DataFrame.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: Containing data in the TSV file.
    """
    return pd.read_csv(
        file_path,
        delimiter="\t",
        index_col=0
    )


def get_concept_codes(
    desc_path: str        = "ncts_sct_rf2/Full/Terminology/sct2_Description_Full-en-au_AU1000036_20250331.txt",
    rel_path : str        = "ncts_sct_rf2/Full/Terminology/sct2_Relationship_Full_AU1000036_20250331.txt",
    group    : str        = CONCEPT_GROUP.PRODUCT_NAME,
    limit    : int | None = None
) -> pd.DataFrame:
    """
    This function reads the description and relationship tables from
    SNOMED CT-AU (with AMT) and fetches just the full terminology name
    (not synonym) and the concept id.

    For step by step guide, see notebook at examples/03_snomed_ct_amt.ipynb.

    Args:
        desc_path (str, optional): Path to the TSV containing the descriptions
            table.
        rel_path (str, optional): Path to the TSV containing the relationships
            table.
        group (int, optional): Concept ID of the concept group to filter on.
            Defaults to CONCEPT_GROUP.PRODUCT_NAME.
        limit (int, optional): Limit the returned concepts. Defaults to None.

    Returns:
        pd.DataFrame: Containing columns:
        - "conceptId" being the concept id
        - "term" being the full terminology description
    """
    # Load the tables from TSV files
    descriptions_df  = read_tsv(desc_path)
    relationships_df = read_tsv(rel_path)

    # Merge and filter for the required information
    # We select only full names (not synonym), currently active
    # and remove duplicates.
    df = (
        pd.merge(
            left     = descriptions_df,
            right    = relationships_df,
            left_on  = "conceptId",
            right_on = "sourceId",
            suffixes = ["_desc", "_rel"]
            )
        .query(
            f"active_desc==1"
            f" & typeId_desc==900000000000003001" # Fully specified name
            f" & destinationId=={group}" # Filter by this group
            )
        .drop_duplicates(subset="conceptId")
        [["conceptId", "term"]]
    )

    # Apply limit if required
    if limit is not None:
        df = df.head(limit)

    return df
