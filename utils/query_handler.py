from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime
import os
import re
import difflib
import random
import numpy as np
from scipy.stats import pearsonr
from utils.column_classifier import infer_column_type

class QueryHandler:
    MAX_CONTENT_LINE_LENGTH = 56  # or any value you want
    INTENT_KEYWORDS = {
        'details': ['details', 'show', 'display', 'find', 'get', 'record', 'info', 'information'],
        'summary': ['summary', 'summarize', 'overview', 'highlight', 'key points'],
        'statistics': ['average', 'mean', 'total', 'sum', 'count', 'max', 'min', 'statistics', 'stats', 'pass', 'fail', 'percent', 'who', 'how many'],
        'top': ['top', 'highest', 'best', 'maximum', 'rank', 'ranking'],
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection'],
        'comparison': ['compare', 'versus', 'difference', 'vs'],
        'trend': ['trend', 'over time', 'history', 'pattern'],
        'correlation': ['correlate', 'correlation', 'relationship', 'related'],
        'filter': ['where', 'with', 'having', 'filter'],
        'aggregation': ['group by', 'by', 'per', 'aggregate'],
        'outlier': ['outlier', 'anomaly', 'unusual', 'extreme'],
        'distribution': ['distribution', 'spread', 'range', 'histogram'],
        'missing': ['missing', 'null', 'empty', 'incomplete'],
        'change': ['change', 'shift', 'increase', 'decrease', 'delta'],
        'conditional_aggregation': ['average.*by.*with', 'sum.*by.*with', 'count.*by.*with'],
        'pattern': ['contains', 'includes', 'matches', 'pattern'],
        'sort': ['sort', 'order', 'arrange']
    }
    SUBJECT_SYNONYMS = {
        'maths': 'math', 'math': 'math',
        'physics': 'physics', 'phy': 'physics',
        'chemistry': 'chem', 'chem': 'chem',
        'english': 'eng', 'eng': 'eng',
        'computer': 'computer', 'comp': 'computer',
        'sales': 'sales', 'revenue': 'sales',
        'ticket': 'ticket', 'issue': 'ticket',
        'order': 'order', 'booking': 'order',
        'campaign': 'campaign', 'ad': 'campaign',
        'shipment': 'shipment', 'delivery': 'shipment',
        'property': 'property', 'real_estate': 'property',
        'crop': 'crop', 'yield': 'crop',
        'guest': 'guest', 'reservation': 'guest',
        'vehicle': 'vehicle', 'car': 'vehicle',
        'policy': 'policy', 'claim': 'policy',
        'employee': 'employee', 'staff': 'employee',
        'attendance': 'attendance', 'participation': 'attendance',
        'complaint': 'complaint', 'issue': 'complaint',
        'region': 'region', 'area': 'region'
    }
    DOMAIN_CONTEXT = {
        'education': {'metric': 'score', 'entity': 'student', 'key_field': 'grade', 'time_field': 'date'},
        'customer_support': {'metric': 'resolution_time', 'entity': 'customer', 'key_field': 'ticket', 'time_field': 'created_at'},
        'retail': {'metric': 'sales', 'entity': 'product', 'key_field': 'order', 'time_field': 'order_date'},
        'marketing': {'metric': 'conversion', 'entity': 'campaign', 'key_field': 'lead', 'time_field': 'date'},
        'gaming': {'metric': 'score', 'entity': 'player', 'key_field': 'level', 'time_field': 'played_at'},
        'legal': {'metric': 'case_duration', 'entity': 'client', 'key_field': 'case', 'time_field': 'filed_date'},
        'logistics': {'metric': 'delivery_time', 'entity': 'shipment', 'key_field': 'route', 'time_field': 'ship_date'},
        'manufacturing': {'metric': 'output', 'entity': 'machine', 'key_field': 'unit', 'time_field': 'production_date'},
        'real_estate': {'metric': 'price', 'entity': 'property', 'key_field': 'sale', 'time_field': 'listed_date'},
        'agriculture': {'metric': 'yield', 'entity': 'crop', 'key_field': 'harvest', 'time_field': 'harvest_date'},
        'energy': {'metric': 'consumption', 'entity': 'facility', 'key_field': 'usage', 'time_field': 'recorded_at'},
        'hospitality': {'metric': 'booking_rate', 'entity': 'guest', 'key_field': 'reservation', 'time_field': 'check_in'},
        'automobile': {'metric': 'sales', 'entity': 'vehicle', 'key_field': 'model', 'time_field': 'sale_date'},
        'telecommunications': {'metric': 'data_usage', 'entity': 'subscriber', 'key_field': 'plan', 'time_field': 'usage_date'},
        'government': {'metric': 'budget', 'entity': 'project', 'key_field': 'policy', 'time_field': 'approved_date'},
        'food_beverage': {'metric': 'sales', 'entity': 'order', 'key_field': 'dish', 'time_field': 'order_date'},
        'it_services': {'metric': 'uptime', 'entity': 'server', 'key_field': 'project', 'time_field': 'logged_at'},
        'event_management': {'metric': 'attendance', 'entity': 'event', 'key_field': 'booking', 'time_field': 'event_date'},
        'insurance': {'metric': 'premium', 'entity': 'policy', 'key_field': 'claim', 'time_field': 'issued_date'},
        'hr_resources': {'metric': 'performance', 'entity': 'employee', 'key_field': 'salary', 'time_field': 'review_date'},
        'generic': {'metric': 'value', 'entity': 'item', 'key_field': 'id', 'time_field': 'date'}
    }

    def __init__(self, logger=None, config: Optional[dict] = None):
        self.logger = logger or self._create_simple_logger()
        self.model = None
        self.query_history = []
        self.previous_result = None

    def _create_simple_logger(self):
        class SimpleLogger:
            def log_info(self, msg): print(f"[INFO] {msg}")
            def log_warning(self, msg): print(f"[WARNING] {msg}")
            def log_error(self, msg): print(f"[ERROR] {msg}")
            def info(self, msg): self.log_info(msg)
            def warning(self, msg): self.log_warning(msg)
            def error(self, msg): self.log_error(msg)
        return SimpleLogger()
    def _find_matching_entity(self, query: str, data: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Finds best matching ID or Name from the dataset using fuzzy logic."""
        from fuzzywuzzy import fuzz

        query_lower = query.lower()
        text_cols = data.select_dtypes(include='object').columns

        best_score = 0
        best_value = None
        best_column = None

        for col in text_cols:
            for val in data[col].dropna().astype(str).unique():
                val_lower = val.lower()
                score = fuzz.partial_ratio(val_lower, query_lower)
                if score > best_score and score >= 80:
                    best_score = score
                    best_value = val
                    best_column = col

        return best_value, best_column

    def get_relevant_fields(self, query: str, columns: List[str]) -> List[str]:
        query_words = re.findall(r'\w+', query.lower())
        matched = []
        for col in columns:
            col_l = col.lower()
            if any(word in col_l for word in query_words):
                matched.append(col)
            elif any(difflib.get_close_matches(word, [col_l], cutoff=0.75) for word in query_words):
                matched.append(col)
        return list(set(matched))


    def parse_filter_conditions(self, query: str, columns: list) -> List[Tuple[str, str, float]]:
        conditions = []
        query_lower = query.lower()
        for col in columns:
            if infer_column_type(col) in ["metric", "subject", "attendance"]:
                matches = re.findall(rf"{col.lower().replace('_', ' ')}\s*(>|>=|<|<=|=)\s*(\d+\.?\d*)", query_lower)
                for op, val in matches:
                    conditions.append((col, op, float(val)))
        return conditions
    def suggest_fields_from_query(self, question: str, all_columns: list, top_n: Optional[int] = None) -> list:
        question_words = re.findall(r'\w+', question.lower())
        scored = []

        for col in all_columns:
            col_clean = col.lower().replace("_", " ")
            score = sum(1 for word in question_words if word in col_clean)
            if score > 0:
                scored.append((col, score))
            else:
                for word in question_words:
                    if difflib.get_close_matches(word, col_clean.split(), cutoff=0.8):
                        scored.append((col, 1))
                        break

        scored = sorted(scored, key=lambda x: -x[1])

        if top_n is not None and top_n > 0:
            return [col for col, _ in scored[:top_n]]
        else:
            return [col for col, _ in scored]

    def suggest_next_questions(self, data: pd.DataFrame, query: str, domain: str) -> List[str]:
        columns = data.columns
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        id_col = self.detect_id_column(data)
        time_cols = [col for col in columns if infer_column_type(col) == "time"]
        category_cols = [col for col in columns if infer_column_type(col) == "category"]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        suggestions = []

        if numeric_cols:
            col = random.choice(numeric_cols)
            suggestions.extend([
                f"What is the average {col.replace('_', ' ')} for all {context['entity']}s?",
                f"Who has the highest {col.replace('_', ' ')}?",
                f"Show {context['entity']}s with {col.replace('_', ' ')} > {data[col].mean():.0f}.",
                f"Show outliers in {col.replace('_', ' ')}.",
                f"Show distribution of {col.replace('_', ' ')}."
            ])
        if id_col and len(data) > 0:
            sample_id = data[id_col].iloc[0]
            suggestions.append(f"Show details of {sample_id}.")
        if time_cols:
            suggestions.append(f"What is the trend of {context['metric']} over time?")
            suggestions.append(f"Show changes in {context['metric']}.")
        if category_cols:
            col = random.choice(category_cols)
            suggestions.append(f"What is the average {context['metric']} by {col.replace('_', ' ')}?")
            suggestions.append(f"Show {context['entity']}s where {col.replace('_', ' ')} contains '{data[col].iloc[0]}'.")
        if numeric_cols and len(numeric_cols) > 1:
            suggestions.append(f"What correlates with {random.choice(numeric_cols).replace('_', ' ')}?")
        if self.query_history:
            suggestions.append(f"Refine previous query: '{self.query_history[-1]}'")
        if self.previous_result:
            suggestions.append(f"Sort previous results by {context['metric']} descending.")
            suggestions.append(f"Export previous results to CSV.")

        return random.sample(suggestions, min(self.config['max_suggestions'], len(suggestions)))

    def format_entity_summary(sid, row, fields, all_num_cols):
            name_col = next((col for col in row.index if infer_column_type(col) == "name"), None)
            name = row.get(name_col, "Unknown")
        # Build response parts for each requested field
            parts = []
            used_fields= set()
            # What fields were actually asked for
            explicitly_requested = set(fields)
            for col in fields:
                if col in used_fields or col not in row.index:
                    continue
                used_fields.add(col)
                val = row[col]
                role = infer_column_type(col)
                pretty = col.replace("_", " ").title()
                if pd.isna(val):
                    parts.append(f"has no data for {pretty}")
                elif role == "subject":
                    parts.append(f"scored {val} in {pretty}")
                elif role == "age":
                    parts.append(f"is {val} years old")
                elif role == "gender":
                    parts.append(f"is identified as {val}")
                elif role == "attendance":
                    parts.append(f"has an attendance of {val}%")
                elif role == "extracurricular":
                        if str(val).strip().lower() in ["yes", "1", "true"]:
                            parts.append("participated in extracurricular activities")
                        else:
                            parts.append("did not participate in extracurricular activities")
                elif "total" in col.lower() or "sum" in col.lower():
                    parts.append(f"has a total of **{val}** marks")
                else:
                    parts.append(f"has {pretty} as {val}")
            if not parts:
                return f"❌ **Error**: No valid data found for the requested fields for ID **{sid}**."
        
            return f"📊 **Student {sid}** ({name}): " + ", ".join(parts) + "."

    def classify_intent(self, query: str) -> tuple[str, float, List[str]]:
        query_lower = query.lower()
        query_words = re.findall(r'\w+', query_lower)

        intent_scores = {intent: 0 for intent in self.INTENT_KEYWORDS}

        # 🔹 Priority overrides
        if 'pass percentage' in query_lower or 'percent passed' in query_lower:
            intent_scores['statistics'] += 3
        if 'how many failed' in query_lower or 'who failed' in query_lower:
            intent_scores['statistics'] += 3
        if 'average' in query_lower or 'mean' in query_lower:
            intent_scores['statistics'] += 2
        if 'predict' in query_lower or 'forecast' in query_lower:
            intent_scores['prediction'] += 3
        if 'compare' in query_lower or 'versus' in query_lower:
            intent_scores['comparison'] += 3
        if 'trend' in query_lower or 'over time' in query_lower:
            intent_scores['trend'] += 3
        if 'correlate' in query_lower or 'relationship' in query_lower:
            intent_scores['correlation'] += 3
        if any(op in query_lower for op in ['>', '<', '>=', '<=', '=']) and 'with' in query_lower:
            intent_scores['filter'] += 3
        if 'by' in query_lower and any(agg in query_lower for agg in ['average', 'sum', 'count']):
            intent_scores['aggregation'] += 3
        if 'with' in query_lower:
            intent_scores['conditional_aggregation'] += 4
        if 'outlier' in query_lower or 'anomaly' in query_lower:
            intent_scores['outlier'] += 3
        if 'distribution' in query_lower or 'spread' in query_lower:
            intent_scores['distribution'] += 3
        if 'missing' in query_lower or 'null' in query_lower:
            intent_scores['missing'] += 3
        if 'change' in query_lower or 'delta' in query_lower:
            intent_scores['change'] += 3
        if 'contains' in query_lower or 'matches' in query_lower:
            intent_scores['pattern'] += 3
        if 'sort' in query_lower or 'order' in query_lower:
            intent_scores['sort'] += 3

        # 🔹 NEW: Quantitative filters
        if re.search(r'(score|scored|mark[s]?)\s+(above|over|greater than|more than|>=)\s+\d+', query_lower):
            intent_scores['filter'] += 4
        elif re.search(r'(score|scored|mark[s]?)\s+(below|under|less than|<)\s+\d+', query_lower):
            intent_scores['filter'] += 4
        elif re.search(r'\b(score|marks?)\s*[><=]+\s*\d+', query_lower):
            intent_scores['filter'] += 3
        elif re.search(r'\b(list|show|display)\s+.*(above|below|greater|less|over|under)\s+\d+', query_lower):
            intent_scores['filter'] += 3

        # 🔹 Generic student/record-level query
        if re.search(r'\b[a-zA-Z0-9]{3,}\b', query_lower) and any(word in query_lower for word in ['physics', 'computer', 'math', 'sales']):
            intent_scores['details'] += 3

        # 🔹 Keyword-based boosting
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if intent == 'conditional_aggregation':
                continue
            for word in query_words:
                if word in keywords:
                    intent_scores[intent] += 1

        # 🔹 Pick best intent
        max_intent = max(intent_scores, key=intent_scores.get)
        confidence = min((intent_scores[max_intent] / max(1, len(query_words))), 0.95) if query_words else 0.3

        # 🔹 Extract subject-like tokens (excluding known keywords)
        flat_keywords = set(sum(self.INTENT_KEYWORDS.values(), []))
        detected_subjects = [
            self.SUBJECT_SYNONYMS.get(w, w)
            for w in query_words if w not in flat_keywords and len(w) > 2
        ]

        return max_intent, confidence, detected_subjects

    def detect_id_column(self, data: pd.DataFrame) -> Optional[str]:
        if data is None or data.empty:
            return None
        id_patterns = [r'.*id.*', r'.*key.*', r'.*code.*', r'.*number.*', r'.*no.*', r'.*ticket.*', r'.*order.*']
        for pattern in id_patterns:
            for col in data.columns:
                if re.match(pattern, col.lower()):
                    return col
        return data.columns[0] if data.columns.size > 0 else None

    def _format_professional_response(self, title: str, content: str, response_type: str = "info") -> str:
        icons = {
            'success': '✅', 'info': 'ℹ️', 'warning': '⚠️', 'error': '❌',
            'details': '📋', 'summary': '📊', 'statistics': '📈', 'top': '🏆',
            'prediction': '🔮', 'comparison': '⚖️', 'trend': '📅',
            'correlation': '🔗', 'filter': '🔍', 'aggregation': '📑',
            'outlier': '⚠️', 'distribution': '📊', 'missing': '❓',
            'change': '🔄', 'conditional_aggregation': '📑', 'pattern': '🔎', 'sort': '📇'
        }
        icon = icons.get(response_type, 'ℹ️')
        content_lines = content.split('\n')
        formatted_lines = []
        for line in content_lines:
            if len(line) > self.MAX_CONTENT_LINE_LENGTH:
                line = line[:self.MAX_CONTENT_LINE_LENGTH - 3] + '...'
            formatted_lines.append(line)
            return content
   
    def handle_query(self, user_id: str, query: str, data: pd.DataFrame, model=None, domain: str = 'generic') -> dict:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return self.error_response("Invalid or empty dataset")

        try:
            # Step 1: Understand the intent
            intent, confidence, detected_subjects = self.classify_intent(query)

            if confidence < 0.25:
                return self.warning_response("Low Confidence",
                                          f"I'm not confident about the intent of your query. Detected: **{intent}** ({confidence:.2f})")

            # Step 2: Setup available handlers
            handler_map = {
                "statistics": self.handle_statistics_query,
                "summary": self.handle_summary_query,
                "details": self.handle_details_query,
                "prediction": self.handle_prediction_query,
                "comparison": self.handle_comparison_query,
                "trend": self.handle_trend_query,
                "correlation": self.handle_correlation_query,
                "filter": self.handle_filter_query,
                "aggregation": self.handle_aggregation_query,
                "conditional_aggregation": self.handle_conditional_aggregation_query,
                "missing": self.handle_missing_query,
                "outlier": self.handle_outlier_query,
                "distribution": self.handle_distribution_query,
                "change": self.handle_change_query,
                "pattern": self.handle_pattern_query,
                "sort": self.handle_sort_query,
                "general": self.handle_general_query
            }

            handler = handler_map.get(intent)
            if not handler:
                return self.error_response("Unsupported intent",
                                            f"❌ No handler implemented for intent: `{intent}`")

            # Step 3: Call the handler function
            return handler(
                user_id=user_id,
                query=query,
                data=data,
                detected_subjects=detected_subjects,
                domain=domain,
            )

        except Exception as e:
            return self.error_response("Query handling failed", str(e))

    def handle_details_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        matched_value, id_col = self._find_matching_entity(query, data)
        if not matched_value:
            return self.error_response("Entity Not Found", "Could not find a matching student ID or name.")

        row = data[data[id_col].astype(str).str.lower() == matched_value.lower()]
        if row.empty:
            return self.error_response("Record Missing", f"No record found for `{matched_value}`.")

        relevant_fields = self.get_relevant_fields(query, data.columns)
        if not relevant_fields:
            relevant_fields = [col for col in data.columns if infer_column_type(col) in ["subject", "percentage"]]

        summary = self.format_entity_summary(
            entity_id=matched_value,
            row=row.iloc[0],
            relevant_fields=relevant_fields,
            domain=domain
        )

        return {'success': True, 'message': summary, 'type': 'details'}

    def format_entity_summary(self, entity_id: str, row: pd.Series, relevant_fields: List[str], domain: str = "generic") -> str:
        name_col = next((col for col in row.index if infer_column_type(col) == "name"), None)
        name = row.get(name_col, entity_id)

        scores = []
        extra = []

        for field in relevant_fields:
            if field not in row or pd.isna(row[field]):
                continue

            value = row[field]
            role = infer_column_type(field)
            label = field.replace("_", " ").title()

            if role == "subject":
                scores.append(f"{value} in {label}")
            elif role == "percentage":
                extra.append(f"with an overall percentage of {value}%")
            elif role == "attendance":
                extra.append(f"and an attendance of {value}%")
            elif role == "total" in field.lower():
                extra.append(f"with total marks of {value}")
            else:
                extra.append(f"{label}: {value}")

        # Compose score sentence
        if scores:
            if len(scores) == 1:
                score_sentence = f"{name} scored {scores[0]}"
            else:
                score_sentence = f"{name} scored " + ", ".join(scores[:-1]) + f", and {scores[-1]}"
        else:
            score_sentence = f"{name}'s requested scores could not be found"

        # Combine extra details if any
        if extra:
            score_sentence += " " + " ".join(extra)

        return score_sentence + "."


    def handle_statistics_query(
        self,
        user_id: str,
        query: str,
        data: pd.DataFrame,
        detected_subjects: List[str],
        domain: str,
        pass_threshold: Optional[int] = None
    ) -> dict:
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for statistics.")

        query_lower = query.lower()
        target_cols = self.get_relevant_fields(query, numeric_cols) or numeric_cols
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT.get('generic', {'entity': 'record'}))
        content_lines = []

        # Determine threshold
        pass_mark = pass_threshold if pass_threshold is not None else 40  # default for education
        name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)

        for col in target_cols:
            col_name = col.replace('_', ' ').title()
            if 'pass percentage' in query_lower or 'percent passed' in query_lower:
                passed = data[data[col] >= pass_mark]
                percent = (len(passed) / len(data)) * 100 if len(data) > 0 else 0
                content_lines.append(f"{percent:.2f}% of {context['entity']}s passed in {col_name}.")

            elif 'how many failed' in query_lower:
                failed = data[data[col] < pass_mark]
                content_lines.append(f"{len(failed)} {context['entity']}s failed in {col_name}.")

            elif 'who failed' in query_lower:
                failed = data[data[col] < pass_mark]
                if failed.empty:
                    content_lines.append(f"No {context['entity']}s failed in {col_name}.")
                else:
                    for _, row in failed.iterrows():
                        name = row.get(name_col) if name_col else row[self.detect_id_column(data)]
                        content_lines.append(f"{name} failed with a {col_name} of {row[col]:.2f}.")

            elif 'average' in query_lower or 'mean' in query_lower:
                content_lines.append(f"The average {col_name} is {data[col].mean():.2f}.")

            elif 'max' in query_lower or 'highest' in query_lower:
                content_lines.append(f"The highest {col_name} is {data[col].max():.2f}.")

            elif 'min' in query_lower or 'lowest' in query_lower:
                content_lines.append(f"The lowest {col_name} is {data[col].min():.2f}.")

            else:
                stats = data[col].describe()
                content_lines.extend([
                    f"📊 Statistics for {col_name}:",
                    f"- Count: {int(stats['count'])}",
                    f"- Mean: {stats['mean']:.2f}",
                    f"- Min: {stats['min']:.2f}",
                    f"- Max: {stats['max']:.2f}"
                ])
            content_lines.append("")

        formatted_response = self._format_professional_response(
            "📈 Statistical Analysis", "\n".join(content_lines), response_type="statistics"
        )
        return {'type': 'statistics', 'message': formatted_response, 'success': True}

    def handle_top_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        numbers = re.findall(r'\d+', query)
        top_n = int(numbers[0]) if numbers else self.config['top_n']
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for ranking.")
        
        target_cols = self.get_relevant_fields(query, numeric_cols)
        sort_column = target_cols[0] if target_cols else numeric_cols[0]
        top_records = data.nlargest(top_n, sort_column)
        id_column = self.detect_id_column(data)
        name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        content_lines = [f"The top {top_n} {context['entity']}s in {sort_column.replace('_', ' ').title()} are:"]
        for i, (_, row) in enumerate(top_records.iterrows(), 1):
            id_val = row[id_column] if id_column else f"Record {i}"
            name = row[name_col] if name_col else id_val
            content_lines.append(f"{i}. {name} with {row[sort_column]}.")
        
        formatted_response = self._format_professional_response(
            f"Top {top_n} Rankings", "\n".join(content_lines), "top"
        )
        return {'type': 'top', 'message': formatted_response, 'success': True, 'data': top_records.to_dict('records')}

    def handle_summary_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        summary_lines = [
            f"The dataset contains {data.shape[0]:,} {context['entity']} records with {data.shape[1]} attributes.",
            ""
        ]
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            summary_lines.append(f"Key {context['metric']} insights:")
            for col in numeric_cols[:3]:
                stats = data[col].describe()
                summary_lines.extend([
                    f"- {col.replace('_', ' ').title()}:",
                    f"  Mean: {stats['mean']:.2f}",
                    f"  Range: {stats['min']:.2f} to {stats['max']:.2f}"
                ])
        
        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Summary", "\n".join(summary_lines), "summary"
        )
        return {'type': 'summary', 'message': formatted_response, 'success': True}

    def handle_prediction_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        if self.model is None:
            return self.warning_response("No Model", "No trained model available for predictions.")
        
        query_lower = query.lower()
        id_column = self.detect_id_column(data)
        possible_ids = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query_lower)
        target_cols = self.get_relevant_fields(query, data.select_dtypes(include='number').columns.tolist())
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])

        if not possible_ids or not target_cols:
            return self.warning_response("Invalid Query", "Please specify an ID and a metric to predict (e.g., 'Predict sales for CUST001').")

        target_col = target_cols[0]
        normalized_col = data[id_column].astype(str).str.lower()
        matched_row = data[normalized_col.isin(possible_ids)]
        if matched_row.empty:
            return self.warning_response("Record Not Found", f"No record found for ID: {possible_ids[0]}")

        feature_cols = [col for col in data.select_dtypes(include='number').columns if col != target_col]
        if not feature_cols:
            return self.warning_response("No Features", "No numeric features available for prediction.")

        X = matched_row[feature_cols].values
        try:
            prediction = self.model.predict(X)[0]
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = max(proba) * 100
                content = f"The predicted {target_col.replace('_', ' ')} for {possible_ids[0]} is {prediction:.2f} with {confidence:.1f}% confidence."
            else:
                content = f"The predicted {target_col.replace('_', ' ')} for {possible_ids[0]} is {prediction:.2f}."
        except Exception as e:
            return self.warning_response("Prediction Error", f"Failed to make prediction: {str(e)}")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Prediction", content, "prediction"
        )
        return {'type': 'prediction', 'message': formatted_response, 'success': True}

    def handle_comparison_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        id_column = self.detect_id_column(data)
        if not id_column:
            return self.warning_response("No ID Column", "Could not identify ID column in dataset.")
        
        query_lower = query.lower()
        possible_ids = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query_lower)
        if len(possible_ids) < 2:
            return self.warning_response("Invalid Query", "Please specify at least two IDs to compare (e.g., 'Compare CUST001 and CUST002').")
        
        normalized_col = data[id_column].astype(str).str.lower()
        matched_rows = data[normalized_col.isin(possible_ids)]
        if len(matched_rows) < 2:
            return self.warning_response("Records Not Found", f"Not enough records found for IDs: {', '.join(possible_ids)}")
        
        target_cols = self.get_relevant_fields(query, data.select_dtypes(include='number').columns.tolist())
        target_col = target_cols[0] if target_cols else data.select_dtypes(include='number').columns[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        content_lines = [f"Comparing {context['entity']}s in {target_col.replace('_', ' ').title()}:"]
        
        for _, row in matched_rows.iterrows():
            id_val = row[id_column]
            name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
            name = row[name_col] if name_col else id_val
            content_lines.append(f"- {name} ({id_val}): {row[target_col]}")
        
        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Comparison", "\n".join(content_lines), "comparison"
        )
        return {'type': 'comparison', 'message': formatted_response, 'success': True, 'data': matched_rows.to_dict('records')}

    def handle_trend_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        time_cols = [col for col in data.columns if infer_column_type(col) == "time"]
        if not time_cols:
            return self.warning_response("No Time Data", "No time-based columns available for trend analysis.")
        
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for trend analysis.")
        
        time_col = time_cols[0]
        target_cols = self.get_relevant_fields(query, numeric_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])

        try:
            data[time_col] = pd.to_datetime(data[time_col])
            grouped = data.groupby(data[time_col].dt.to_period('M'))[target_col].mean().reset_index()
            content_lines = [f"Monthly trend of {target_col.replace('_', ' ').title()} for {context['entity']}s:"]
            for _, row in grouped.iterrows():
                content_lines.append(f"- {row[time_col].strftime('%Y-%m')}: {row[target_col]:.2f}")
        except Exception as e:
            return self.warning_response("Trend Error", f"Failed to analyze trend: {str(e)}")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Trend", "\n".join(content_lines), "trend"
        )
        return {'type': 'trend', 'message': formatted_response, 'success': True}

    def handle_correlation_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            return self.warning_response("Insufficient Data", "Need at least two numeric columns for correlation analysis.")
        
        target_cols = self.get_relevant_fields(query, numeric_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        content_lines = [f"Correlations with {target_col.replace('_', ' ').title()}:"]
        
        for col in numeric_cols:
            if col != target_col:
                try:
                    corr, _ = pearsonr(data[target_col].dropna(), data[col].dropna())
                    if abs(corr) >= self.config['correlation_threshold']:
                        content_lines.append(f"- {col.replace('_', ' ').title()}: {corr:.2f}")
                except:
                    continue
        
        if len(content_lines) == 1:
            content_lines.append("No significant correlations found.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Correlations", "\n".join(content_lines), "correlation"
        )
        return {'type': 'correlation', 'message': formatted_response, 'success': True}

    def handle_filter_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        conditions = self.parse_filter_conditions(query, data.columns)
        if not conditions:
            return self.warning_response("Invalid Filter", "No valid filter conditions found (e.g., 'Show students with Maths > 80').")
        
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        filtered_data = data.copy()
        for col, op, val in conditions:
            if op == '>':
                filtered_data = filtered_data[filtered_data[col] > val]
            elif op == '>=':
                filtered_data = filtered_data[filtered_data[col] >= val]
            elif op == '<':
                filtered_data = filtered_data[filtered_data[col] < val]
            elif op == '<=':
                filtered_data = filtered_data[filtered_data[col] <= val]
            elif op == '=':
                filtered_data = filtered_data[filtered_data[col] == val]

        if filtered_data.empty:
            return self.warning_response("No Results", "No records match the filter conditions.")
        
        id_column = self.detect_id_column(data)
        name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
        content_lines = [f"{context['entity'].title()}s matching the filter:"]
        for _, row in filtered_data.head(5).iterrows():
            id_val = row[id_column] if id_column else "Unknown"
            name = row[name_col] if name_col else id_val
            content_lines.append(f"- {name} ({id_val})")

        if len(filtered_data) > 5:
            content_lines.append(f"...and {len(filtered_data) - 5} more.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Filter Results", "\n".join(content_lines), "filter"
        )
        return {'type': 'filter', 'message': formatted_response, 'success': True, 'data': filtered_data.to_dict('records')}

    def handle_aggregation_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        category_cols = [col for col in data.columns if infer_column_type(col) == "category"]
        if not category_cols:
            return self.warning_response("No Categories", "No categorical columns available for aggregation.")
        
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for aggregation.")
        
        query_lower = query.lower()
        target_cols = self.get_relevant_fields(query, numeric_cols)
        category_cols = self.get_relevant_fields(query, category_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        category_col = category_cols[0] if category_cols else category_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        agg_func = 'mean' if 'average' in query_lower else 'sum' if 'total' in query_lower else 'count'
        grouped = data.groupby(category_col)[target_col].agg(agg_func).reset_index()
        content_lines = [f"{agg_func.title()} {target_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}:"]
        for _, row in grouped.iterrows():
            content_lines.append(f"- {row[category_col]}: {row[target_col]:.2f}")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Aggregation", "\n".join(content_lines), "aggregation"
        )
        return {'type': 'aggregation', 'message': formatted_response, 'success': True, 'data': grouped.to_dict('records')}

    def handle_outlier_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for outlier detection.")
        
        query_lower = query.lower()
        target_cols = self.get_relevant_fields(query, numeric_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        Q1 = data[target_col].quantile(0.25)
        Q3 = data[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config['outlier_threshold'] * IQR
        upper_bound = Q3 + self.config['outlier_threshold'] * IQR
        outliers = data[(data[target_col] < lower_bound) | (data[target_col] > upper_bound)]
        
        if outliers.empty:
            content_lines = [f"No outliers found in {target_col.replace('_', ' ').title()}."]
        else:
            id_column = self.detect_id_column(data)
            name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
            content_lines = [f"Outliers in {target_col.replace('_', ' ').title()}:"]
            for _, row in outliers.iterrows():
                id_val = row[id_column] if id_column else "Unknown"
                name = row[name_col] if name_col else id_val
                content_lines.append(f"- {name} ({id_val}): {row[target_col]}")
        
        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Outliers", "\n".join(content_lines), "outlier"
        )
        return {'type': 'outlier', 'message': formatted_response, 'success': True, 'data': outliers.to_dict('records')}

    def handle_distribution_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for distribution analysis.")
        
        query_lower = query.lower()
        target_cols = self.get_relevant_fields(query, numeric_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        stats = data[target_col].describe()
        content_lines = [
            f"Distribution of {target_col.replace('_', ' ').title()}:",
            f"- Count: {stats['count']:.0f}",
            f"- Mean: {stats['mean']:.2f}",
            f"- Std Dev: {stats['std']:.2f}",
            f"- Min: {stats['min']:.2f}",
            f"- 25%: {stats['25%']:.2f}",
            f"- 50% (Median): {stats['50%']:.2f}",
            f"- 75%: {stats['75%']:.2f}",
            f"- Max: {stats['max']:.2f}"
        ]

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Distribution", "\n".join(content_lines), "distribution"
        )
        return {'type': 'distribution', 'message': formatted_response, 'success': True}

    def handle_missing_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        missing_counts = data.isnull().sum()
        content_lines = [f"Missing data in the {context['entity']} dataset:"]
        any_missing = False
        for col, count in missing_counts.items():
            if count > 0:
                content_lines.append(f"- {col.replace('_', ' ').title()}: {count} missing values")
                any_missing = True
        
        if not any_missing:
            content_lines.append("No missing values found.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Missing Data", "\n".join(content_lines), "missing"
        )
        return {'type': 'missing', 'message': formatted_response, 'success': True}

    def handle_change_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        time_cols = [col for col in data.columns if infer_column_type(col) == "time"]
        if not time_cols:
            return self.warning_response("No Time Data", "No time-based columns available for change detection.")
        
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for change detection.")
        
        time_col = time_cols[0]
        target_cols = self.get_relevant_fields(query, numeric_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])

        try:
            data[time_col] = pd.to_datetime(data[time_col])
            grouped = data.groupby(data[time_col].dt.to_period('M'))[target_col].mean().reset_index()
            changes = grouped[target_col].pct_change().fillna(0)
            content_lines = [f"Monthly changes in {target_col.replace('_', ' ').title()} for {context['entity']}s:"]
            for i, (_, row) in enumerate(grouped.iterrows()):
                if i == 0:
                    continue
                change = changes[i]
                if abs(change) >= self.config['change_threshold']:
                    content_lines.append(f"- {row[time_col].strftime('%Y-%m')}: {change*100:.2f}% change")
        except Exception as e:
            return self.warning_response("Change Error", f"Failed to analyze changes: {str(e)}")

        if len(content_lines) == 1:
            content_lines.append("No significant changes detected.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Changes", "\n".join(content_lines), "change"
        )
        return {'type': 'change', 'message': formatted_response, 'success': True}

    def handle_conditional_aggregation_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        category_cols = [col for col in data.columns if infer_column_type(col) == "category"]
        if not category_cols:
            return self.warning_response("No Categories", "No categorical columns available for conditional aggregation.")
        
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            return self.warning_response("No Numeric Data", "No numeric columns available for conditional aggregation.")
        
        conditions = self.parse_filter_conditions(query, numeric_cols)
        if not conditions:
            return self.warning_response("Invalid Filter", "No valid filter conditions found (e.g., 'Average sales by region with sales > 1000').")
        
        query_lower = query.lower()
        target_cols = self.get_relevant_fields(query, numeric_cols)
        category_cols = self.get_relevant_fields(query, category_cols)
        target_col = target_cols[0] if target_cols else numeric_cols[0]
        category_col = category_cols[0] if category_cols else category_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        filtered_data = data.copy()
        for col, op, val in conditions:
            if op == '>':
                filtered_data = filtered_data[filtered_data[col] > val]
            elif op == '>=':
                filtered_data = filtered_data[filtered_data[col] >= val]
            elif op == '<':
                filtered_data = filtered_data[filtered_data[col] < val]
            elif op == '<=':
                filtered_data = filtered_data[filtered_data[col] <= val]
            elif op == '=':
                filtered_data = filtered_data[filtered_data[col] == val]
        
        if filtered_data.empty:
            return self.warning_response("No Results", "No records match the filter conditions for aggregation.")
        
        agg_func = 'mean' if 'average' in query_lower else 'sum' if 'total' in query_lower else 'count'
        grouped = filtered_data.groupby(category_col)[target_col].agg(agg_func).reset_index()
        content_lines = [f"{agg_func.title()} {target_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()} with filter:"]
        for _, row in grouped.iterrows():
            content_lines.append(f"- {row[category_col]}: {row[target_col]:.2f}")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Conditional Aggregation", "\n".join(content_lines), "conditional_aggregation"
        )
        return {'type': 'conditional_aggregation', 'message': formatted_response, 'success': True, 'data': grouped.to_dict('records')}

    def handle_pattern_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        category_cols = [col for col in data.columns if infer_column_type(col) == "category"]
        if not category_cols:
            return self.warning_response("No Categories", "No categorical columns available for pattern matching.")
        
        query_lower = query.lower()
        pattern = re.search(r"contains\s+['\"]?(\w+)['\"]?", query_lower)
        if not pattern:
            return self.warning_response("Invalid Pattern", "Please specify a pattern to match (e.g., 'Show customers with complaints containing delay').")
        
        pattern_str = pattern.group(1)
        target_cols = self.get_relevant_fields(query, category_cols)
        target_col = target_cols[0] if target_cols else category_cols[0]
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        matched_data = data[data[target_col].str.contains(pattern_str, case=False, na=False)]
        if matched_data.empty:
            return self.warning_response("No Matches", f"No {context['entity']}s found with {target_col.replace('_', ' ')} containing '{pattern_str}'.")
        
        id_column = self.detect_id_column(data)
        name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
        content_lines = [f"{context['entity'].title()}s with {target_col.replace('_', ' ').title()} containing '{pattern_str}':"]
        for _, row in matched_data.head(5).iterrows():
            id_val = row[id_column] if id_column else "Unknown"
            name = row[name_col] if name_col else id_val
            content_lines.append(f"- {name} ({id_val}): {row[target_col]}")
        
        if len(matched_data) > 5:
            content_lines.append(f"...and {len(matched_data) - 5} more.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Pattern Matches", "\n".join(content_lines), "pattern"
        )
        return {'type': 'pattern', 'message': formatted_response, 'success': True, 'data': matched_data.to_dict('records')}

    def handle_sort_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        target_cols = self.get_relevant_fields(query, data.columns)
        if not target_cols:
            return self.warning_response("Invalid Column", "Please specify a column to sort by (e.g., 'Sort students by Maths descending').")
        
        query_lower = query.lower()
        sort_col = target_cols[0]
        ascending = 'descending' not in query_lower
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        
        sorted_data = data.sort_values(by=sort_col, ascending=ascending)
        id_column = self.detect_id_column(data)
        name_col = next((col for col in data.columns if infer_column_type(col) == "name"), None)
        content_lines = [f"{context['entity'].title()}s sorted by {sort_col.replace('_', ' ').title()} {'descending' if not ascending else 'ascending'}:"]
        for _, row in sorted_data.head(5).iterrows():
            id_val = row[id_column] if id_column else "Unknown"
            name = row[name_col] if name_col else id_val
            content_lines.append(f"- {name} ({id_val}): {row[sort_col]}")
        
        if len(sorted_data) > 5:
            content_lines.append(f"...and {len(sorted_data) - 5} more.")

        formatted_response = self._format_professional_response(
            f"{context['entity'].title()} Sorted Results", "\n".join(content_lines), "sort"
        )
        return {'type': 'sort', 'message': formatted_response, 'success': True, 'data': sorted_data.to_dict('records')}

    def handle_general_query(self, user_id: str, query: str, data: pd.DataFrame, detected_subjects: List[str], domain: str) -> dict:
        context = self.DOMAIN_CONTEXT.get(domain, self.DOMAIN_CONTEXT['generic'])
        features = [
            f"🔍 Get {context['entity']} details: 'Show details of [ID]'",
            f"📊 Analyze data: 'Calculate average {context['metric']}'",
            f"📈 Check performance: 'Pass percentage in {context['key_field']}' or 'Who failed in {context['key_field']}'",
            f"🏆 Find leaders: 'Show top 5 by {context['metric']}'",
            f"⚖️ Compare: 'Compare [ID1] and [ID2] in {context['metric']}'",
            f"📅 See trends: 'Show {context['metric']} trend over time'",
            f"🔗 Find relationships: 'What correlates with {context['metric']}?'",
            f"🔍 Filter: 'Show {context['entity']}s with {context['metric']} > [value]'",
            f"📑 Statistic: 'Average {context['metric']} by [category]'",
            f"⚠️ Detect outliers: 'Show outliers in {context['metric']}'",
            f"📊 Check distribution: 'Show distribution of {context['metric']}'",
            f"❓ Find missing: 'Show missing data in {context['metric']}'",
            f"🔄 Detect changes: 'Show changes in {context['metric']}'",
            f"🔎 Search patterns: 'Show {context['entity']}s with {context['key_field']} containing [text]'",
            f"📇 Sort data: 'Sort {context['entity']}s by {context['metric']} descending'"
        ]
        content = "You can ask:\n\n" + "\n".join(features) + \
                  f"\n\nThe dataset has {data.shape[0]} {context['entity']} records."
        formatted_response = self._format_professional_response("Query Assistant", content, "info")
        return {'type': 'general', 'message': formatted_response, 'success': True}

    def warning_response(self, title: str, message: str, response_type: str = "warning") -> dict:
        formatted_response = self._format_professional_response(title, message, response_type)
        return {'type': response_type, 'message': formatted_response, 'success': False}

    def error_response(self, title: str, message: str) -> dict:
        formatted_response = self._format_professional_response(title, message, "error")
        return {'type': 'error', 'message': formatted_response, 'success': False}

    def log_answered_query(self, user_id: str, query: str, response: str, domain: str):
        log_path = "logs/answered_queries.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | {user_id} | {domain} | {query} | {response[:300].replace('\n', ' ')}\n")

    def log_unanswered_query(self, user_id: str, query: str, reason: str = "no_match"):
        log_path = "logs/failed_queries.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | {user_id} | {query} | Reason: {reason}\n")

    def export_results(self, data: List[Dict], filename: str = "query_results.csv"):
        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            return f"Results exported to {filename}."
        except Exception as e:
            return f"Failed to export results: {str(e)}"