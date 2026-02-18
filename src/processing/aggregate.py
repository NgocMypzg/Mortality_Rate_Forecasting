import numpy as np
import pandas as pd

def prepare_level_data(
    df,
    level="global",
    filter=None,
    year_start=2000,
    year_end=2024,
    fill_missing_years=True
):
    """
    Hàm chuẩn bị dữ liệu theo level (global, region và country)
    Trong đó:
      level: {'global', 'region', 'country'}
      filter: tên region hoặc country
      year_start: năm bắt đầu
      year_end: năm kết thúc
      fill_missing_years: True/False
    """
    df2 = df.copy()
    df2["Year"] = pd.to_numeric(df2["Year"], errors="coerce")

    # Filter theo level
    if level == "global":
        pass  # không filter

    elif level == "region":
        if filter is None:
            raise ValueError("level='region' cần truyền filter = tên Region")
        if "Region" not in df2.columns:
            raise ValueError("Thiếu cột 'Region'")
        df2 = df2[df2["Region"] == filter]

    elif level == "country":
        if filter is None:
            raise ValueError("level='country' cần truyền filter = tên Country")
        if "Country" not in df2.columns:
            raise ValueError("Thiếu cột 'Country'")
        df2 = df2[df2["Country"] == filter]

    else:
        raise ValueError("level phải thuộc: 'global', 'region', 'country'")

    # Filter năm
    df2 = df2[df2["Year"].between(year_start, year_end)]

    # Group theo Year
    agg_dict = {"Total_Deaths": "sum"}
    has_pop = "Population" in df2.columns
    if has_pop:
        agg_dict["Population"] = "sum"

    out = df2.groupby("Year", as_index=False).agg(agg_dict)

    # Fill đủ năm
    if fill_missing_years:
        years = pd.DataFrame({"Year": np.arange(year_start, year_end + 1)})
        out = years.merge(out, on="Year", how="left")
        out["Total_Deaths"] = out["Total_Deaths"].fillna(0)
        if has_pop:
            out["Population"] = out["Population"].fillna(0)

    # Mortality Rate / 100,000 dân
    if has_pop:
        out["Mortality_Rate"] = np.where(
            out["Population"] > 0,
            (out["Total_Deaths"] / out["Population"]) * 100000,
            0.0
        )

    return out.sort_values("Year").reset_index(drop=True)