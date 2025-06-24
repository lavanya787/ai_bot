def infer_column_type(col: str) -> str:
    col_l = col.lower()

    # === Shared Generic Roles ===
    if "name" in col_l:
        return "name"
    if any(k in col_l for k in ["roll", "id", "code", "uid", "ticket", "ref", "record"]):
        return "id"
    if "gender" in col_l:
        return "gender"
    if "age" in col_l:
        return "age"
    if any(k in col_l for k in ["percent", "gpa", "cgpa", "rate"]):
        return "percentage"
    if any(k in col_l for k in ["attend", "presence"]):
        return "attendance"
    if any(k in col_l for k in ["dob", "date", "time", "timestamp", "created", "logged", "updated"]):
        return "time"
    if any(k in col_l for k in ["total", "sum", "overall", "final", "metric"]):
        return "metric"
    if any(k in col_l for k in ["score", "mark", "grade", "result", "exam"]):
        return "subject"
    if any(k in col_l for k in ["region", "area", "location", "zone", "sector", "branch"]):
        return "category"

    # === Domain-specific Inference ===
    if any(k in col_l for k in ["crop", "soil", "yield", "rainfall", "ph", "farmer", "acre"]):
        return "agriculture"
    if any(k in col_l for k in ["salary", "designation", "title", "employee", "tenure"]):
        return "hr_resources"
    if any(k in col_l for k in ["customer", "client", "complaint", "support", "agent"]):
        return "customer_support"
    if any(k in col_l for k in ["movie", "show", "rating", "genre", "stream", "viewer"]):
        return "entertainment"
    if any(k in col_l for k in ["game", "level", "xp", "score", "achievement"]):
        return "gaming"
    if any(k in col_l for k in ["case", "law", "court", "verdict", "legal"]):
        return "legal"
    if any(k in col_l for k in ["campaign", "ad", "click", "reach", "impression"]):
        return "marketing"
    if any(k in col_l for k in ["delivery", "shipment", "warehouse", "tracking", "logistics"]):
        return "logistics"
    if any(k in col_l for k in ["plant", "machine", "line", "downtime", "defect"]):
        return "manufacturing"
    if any(k in col_l for k in ["property", "rent", "lease", "broker", "sqft"]):
        return "real_estate"
    if any(k in col_l for k in ["solar", "wind", "energy", "power", "grid"]):
        return "energy"
    if any(k in col_l for k in ["hotel", "booking", "guest", "stay", "checkin"]):
        return "hospitality"
    if any(k in col_l for k in ["car", "vehicle", "automobile", "mileage", "engine"]):
        return "automobile"
    if any(k in col_l for k in ["call", "plan", "data", "provider", "network"]):
        return "telecommunications"
    if any(k in col_l for k in ["gov", "scheme", "citizen", "public", "authority"]):
        return "government"
    if any(k in col_l for k in ["food", "beverage", "menu", "recipe", "calories"]):
        return "food_beverage"
    if any(k in col_l for k in ["server", "api", "it", "software", "infra"]):
        return "it_services"
    if any(k in col_l for k in ["event", "ticket", "organizer", "venue"]):
        return "event_management"
    if any(k in col_l for k in ["policy", "premium", "claim", "insurance"]):
        return "insurance"
    if any(k in col_l for k in ["product", "sku", "sales", "units", "retail"]):
        return "retail"
    if any(k in col_l for k in ["student", "subject", "exam", "class", "attendance", "gpa"]):
        return "education"

    return "generic"
