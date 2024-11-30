from pydantic import BaseModel

from flask_openapi3 import Info
from flask_openapi3 import OpenAPI

from model import LogReg, TreeMod

info = Info(title="book API", version="1.0.0")
app = OpenAPI(__name__, info=info)

class RiskQuery(BaseModel):
    country_of_origin: int
    is_nigeria: int
    is_postal_100001: int
    is_duplicate: int
    is_after_hours: int
    category: str


@app.get("/model", summary="Returns risk of operation being fraudulent")
def get_book(query: RiskQuery):
    """
    Returns risk of operation being fraudulent
    """
    if query.country_of_origin > 2 and query.country_of_origin < 0:
        return {
            "message": f"Error: country_of_origin has an incorrect value of {query.country_of_origin}"
        } 
    vals_to_check_names = ("is_nigeria", "is_postal_100001", "is_duplicate", "is_after_hours")
    vals_to_check = (query.is_nigeria, query.is_postal_100001, query.is_duplicate, query.is_after_hours)
    for i, val in enumerate(vals_to_check):
        if val != 0 and val != 1:
            return {
                "message": f"Error: {vals_to_check_names[i]} has an incorrect value of {val}"
            }
    logreg_model = LogReg.load("logreg")
    tree_model = TreeMod.load("treemod")
    args = [
        query.country_of_origin,
        query.is_nigeria,
        query.is_postal_100001,
        query.is_duplicate,
        query.is_after_hours,
        query.category
    ]
    return {
        "message": "ok",
        "data": [
            {"model": "logreg", "probability": logreg_model.predict(*args)},
            {"model": "tree", "probability": tree_model.predict(*args),}
        ]
    }


if __name__ == "__main__":
    app.run(debug=True)