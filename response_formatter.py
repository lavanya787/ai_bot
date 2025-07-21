import re
from collections import Counter

class ResponseFormatter:
    def __init__(self):
        self.common_misspellings = {
            'laq': 'law',
            'ampere s': 'ampère\'s',
            'ampere': 'ampère',
            'ohm s': 'ohm\'s',
            'ohms': 'ohm\'s',
            'voltge': 'voltage',
            'resistence': 'resistance',
            'kirchoff': 'kirchhoff',
            'kirchhoffs': 'kirchhoff\'s',
            'curent': 'current',
            'electic': 'electric',
            'juntion': 'junction',
            'chrge': 'charge',
            'flw': 'flow',
            'potental': 'potential',
            'potentiel': 'potential',
            'electrosttic': 'electrostatic',
            'newton s': 'newton\'s',
            'newtons': 'newton\'s'
        }

    def _correct_prompt(self, prompt: str) -> str:
        """Correct common misspellings in the prompt, preserving case."""
        prompt_lower = prompt.lower()
        corrections = {}
        for misspelling, correction in self.common_misspellings.items():
            if re.search(r'\b' + re.escape(misspelling) + r'\b', prompt_lower):
                corrections[misspelling] = correction
        corrected = prompt
        for misspelling, correction in corrections.items():
            pattern = re.compile(r'\b' + re.escape(misspelling) + r'\b', re.IGNORECASE)
            corrected = pattern.sub(lambda m: correction if m.group(0).islower() else correction.title(), corrected)
        return corrected

    def _extract_key_terms(self, sentences, prompt):
        """Extract key terms dynamically from sentences and prompt."""
        prompt_terms = set(re.findall(r'\b[\w=<>∑∮/+-]+\b', prompt.lower()))
        all_terms = []
        for sentence in sentences:
            all_terms.extend(re.findall(r'\b[\w=<>∑∮/+-]+\b', sentence.lower()))
        term_counts = Counter(all_terms)
        # Select terms that are frequent, in the prompt, or part of equations
        key_terms = {term for term, count in term_counts.items() if count >= 3 or term in prompt_terms or '=' in term}
        # Add definition indicators dynamically
        definition_indicators = set(re.findall(r'\b(states|defined|relates|law|principle|describes)\b', ' '.join(sentences).lower()))
        key_terms.update(definition_indicators)
        return key_terms

    def _clean_sentence(self, sentence):
        """Clean sentence by removing PDF artifacts and normalizing spaces."""
        if not isinstance(sentence, str) or not sentence.strip():
            return ""
        sentence = re.sub(r'\(cid\d+\)', '', sentence)
        sentence = re.sub(r'[/OC\s\d+R/PieceInfo<.*?>]', '', sentence)
        sentence = re.sub(r'[^\w\s,.!?()-=<>∑∮/+-]', ' ', sentence)
        sentence = re.sub(r'\ languagerequired\s', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if len(sentence) < 20 or sentence.count(' ') < 4 or sentence.lower().startswith('(') or sentence.lower().endswith(')'):
            return ""
        return sentence

    def format_response(self, context_result, task_type, prompt):
        content = context_result.get("content", "No relevant information found in the provided documents.")
        sources = context_result.get("sources", ["unknown"])
        confidence = context_result.get("confidence", 0.0)
        sentences = context_result.get("sentences", [])
        
        # Correct prompt
        corrected_prompt = self._correct_prompt(prompt)
        
        response = ""
        if task_type in ["definition", "law"]:
            response = self._format_definition_response(content, sources, sentences, corrected_prompt)
        elif task_type == "explain":
            response = self._format_explanation_response(content, sources, sentences, corrected_prompt)
        elif task_type == "formula":
            response = self._format_formula_response(content, sources, sentences, corrected_prompt)
        elif task_type == "mcq":
            response = self._format_mcq_response(content, sources, sentences, corrected_prompt)
        elif task_type == "summary":
            response = self._format_summary_response(content, sources, sentences)
        elif task_type == "bullet_points":
            response = self._format_bullet_points_response(content, sources, sentences)
        else:
            response = self._format_definition_response(content, sources, sentences, corrected_prompt)  # Default to definition for law-related queries
        
        if content and content != "No relevant information found in the provided documents.":
            response += f"\n\n📚 *Sources: {', '.join(sources)}*"
            response += f"\n*Confidence: {confidence:.2f}*"
        return response

    def _format_definition_response(self, content, sources, sentences, prompt):
        """Format a definition response with precise sentence selection."""
        key_terms = self._extract_key_terms(sentences, prompt)
        main_term = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', prompt)
        main_term = main_term[0] if main_term else prompt.split()[-1].title()

        # Definition patterns for physical laws
        definition_patterns = [
            r'\b' + re.escape(main_term.lower()) + r'\s*(?:is defined as|states that|relates|is given by|describes)\s*([^.!?]+)',
            r'\b' + re.escape(main_term.lower()) + r'\s*[A-Z]\s*=\s*[A-Z\d\s*/+-.∑∮]+',
            r'\b' + re.escape(main_term.lower()) + r'\s*(?:law|principle)\s*[A-Z\s=/+-.∑∮]+'
        ]
        definition_sentences = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if not cleaned:
                continue
            sentence_lower = cleaned.lower()
            # Penalize irrelevant terms
            irrelevant_terms = ['galvanometer', 'torque', 'moment', 'sensitivity', 'ampère', 'newton'] if 'ohm' in main_term.lower() else ['galvanometer', 'torque', 'moment', 'sensitivity']
            irrelevant_score = sum(2.0 for term in irrelevant_terms if term in sentence_lower and term not in prompt.lower())
            score = sum(2.0 for term in key_terms if term in sentence_lower)
            score += sum(3.0 for phrase in ["is defined as", "states that", "relates", "law", "principle", "describes"] if phrase in sentence_lower)
            score += sum(4.0 for pattern in definition_patterns if re.search(pattern, sentence_lower, re.IGNORECASE))
            score -= irrelevant_score
            if score > 5.0:  # Strict threshold for relevance
                definition_sentences.append((cleaned, score))
        
        # Sort by relevance score and select top sentences
        definition_sentences.sort(key=lambda x: x[1], reverse=True)
        definition_sentences = [s[0] for s in definition_sentences][:3]

        # Extract mathematical equation
        math_eqn = re.findall(r'\b[A-Z]\s*=\s*[A-Z\d\s*/+-.∑∮]+|\oint\s*[A-Za-z\s\\]*\s*\.\s*[A-Za-z\s\\]*\s*=\s*[A-Za-z\d\s*/+-.∑∮]+', content)
        math_eqn = [eq.strip() for eq in math_eqn if len(eq.strip()) > 5 and any(term in eq.lower() for term in key_terms)][:1]

        response = f"📋 **{main_term}**\n\n"
        if definition_sentences:
            response += f"**Definition:** {definition_sentences[0]}\n\n"
            if len(definition_sentences) > 1:
                response += "**Key Points:**\n"
                for i, sent in enumerate(definition_sentences[1:], 1):
                    response += f"{i}. {sent}\n"
        else:
            cleaned_content = self._clean_sentence(content[:200])
            response += f"**Definition:** {cleaned_content if cleaned_content else 'No precise definition found in the provided documents.'}...\n\n"

        if math_eqn:
            response += f"\n**Mathematical Form:**\n\n\\[\n{math_eqn[0]}\n\\]\n"

        # Add example only if highly relevant
        example_sentences = [s for s in sentences if any(phrase in s.lower() for phrase in ["example", "application", "for instance"]) and any(term in s.lower() for term in key_terms)]
        if example_sentences:
            response += f"\n**Example:** {self._clean_sentence(example_sentences[0])}\n"

        return response

    def _format_explanation_response(self, content, sources, sentences, prompt):
        """Format an explanation response with relevant details."""
        response = f"💡 **Explanation of {prompt.title()}**\n\n"
        key_terms = self._extract_key_terms(sentences, prompt)
        
        definition_sentences = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if not cleaned:
                continue
            sentence_lower = cleaned.lower()
            sentence_terms = set(re.findall(r'\b[\w=<>∑∮/+-]+\b', sentence_lower))
            score = 2.0 * len(sentence_terms.intersection(key_terms))
            score += 3.0 * sum(1 for t in sentence_terms if '=' in t)
            score += sum(1.5 for term in prompt.lower().split() if term in sentence_lower)
            score -= sum(2.0 for term in ['galvanometer', 'torque', 'ampère', 'sensitivity', 'newton'] if term in sentence_lower and term not in prompt.lower())
            if score > 5.0:
                definition_sentences.append((cleaned, score))
        
        definition_sentences.sort(key=lambda x: x[1], reverse=True)
        definition_sentences = [s[0] for s in definition_sentences][:4]
        
        if definition_sentences:
            response += f"**Overview:** {definition_sentences[0]}\n\n"
            response += f"**Details:**\n"
            for i, sentence in enumerate(definition_sentences[1:], 1):
                response += f"- {sentence}\n"
        else:
            cleaned_content = self._clean_sentence(content[:500])
            response += f"**Overview:** {cleaned_content if cleaned_content else 'No relevant explanation found in the provided documents.'}\n\n"
        
        math_patterns = re.findall(r'\b[\w]\s*=\s*[\w\d\s/+-∑∮]+|\oint\s*[A-Za-z\s\\]*\s*\.\s*[A-Za-z\s\\]*\s*=\s*[A-Za-z\d\s*/+-.∑∮]+', content)
        math_patterns = [eq.strip() for eq in math_patterns if len(eq.strip()) > 5 and any(term in eq.lower() for term in key_terms)][:2]
        if math_patterns:
            response += "\n**Related Equations:**\n"
            for eq in math_patterns:
                response += f"- \\({eq}\\)\n"
        
        return response

    def _format_formula_response(self, content, sources, sentences, prompt):
        """Format a formula response with the primary equation."""
        response = f"🔢 **Formula for {prompt.title()}**\n\n"
        
        key_terms = self._extract_key_terms(sentences, prompt)
        math_patterns = re.findall(r'\b[\w]\s*=\s*[\w\d\s/+-∑∮]+|\oint\s*[A-Za-z\s\\]*\s*\.\s*[A-Za-z\s\\]*\s*=\s*[A-Za-z\d\s*/+-.∑∮]+', content)
        math_patterns = [eq.strip() for eq in math_patterns if len(eq.strip()) > 5 and any(term in eq.lower() for term in key_terms)][:1]
        response += f"**Mathematical Expression:** \\({math_patterns[0] if math_patterns else 'Not found'}\\)\n\n"
        
        definition_sentences = [self._clean_sentence(s) for s in sentences if self._clean_sentence(s)]
        response += f"**Context:** {definition_sentences[0] if definition_sentences else 'Formula derivation'}\n\n"
        
        if len(definition_sentences) > 1:
            response += f"**Application:** {definition_sentences[1]}\n"
        
        return response

    def _format_mcq_response(self, content, sources, sentences, prompt):
        """Format a multiple-choice question response."""
        response = f"❓ **Multiple Choice Question on {prompt.title()}**\n\n"
        
        definition_sentences = [self._clean_sentence(s) for s in sentences if self._clean_sentence(s)]
        if len(definition_sentences) >= 4:
            response += f"**Question:** {definition_sentences[0]}?\n\n"
            response += f"**Options:**\n"
            response += f"A) {definition_sentences[1]}\n"
            response += f"B) {definition_sentences[2]}\n"
            response += f"C) {definition_sentences[3]}\n"
            response += f"D) None of the above\n\n"
        else:
            response += f"**Question Context:** {self._clean_sentence(content[:300])}\n\n"
        
        return response

    def _format_summary_response(self, content, sources, sentences):
        """Format a summary response."""
        response = f"📄 **Summary**\n\n"
        definition_sentences = [self._clean_sentence(s) for s in sentences if self._clean_sentence(s)]
        summary_text = " ".join(definition_sentences[:3]) if definition_sentences else self._clean_sentence(content[:300])
        response += summary_text[:300] + ("..." if len(summary_text) > 300 else "")
        return response

    def _format_bullet_points_response(self, content, sources, sentences):
        """Format a bullet points response."""
        response = f"📝 **Key Points**\n\n"
        definition_sentences = [self._clean_sentence(s) for s in sentences if self._clean_sentence(s)]
        for i, sentence in enumerate(definition_sentences[:5], 1):
            if sentence:
                response += f"- {sentence}\n"
        return response

    def _format_general_response(self, content, sources, sentences, prompt):
        """Format a general response aligned with the query."""
        response = f"💬 **Answer to {prompt.title()}**\n\n"
        key_terms = self._extract_key_terms(sentences, prompt)
        
        definition_sentences = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if not cleaned:
                continue
            sentence_lower = cleaned.lower()
            sentence_terms = set(re.findall(r'\b[\w=<>∑∮/+-]+\b', sentence_lower))
            score = 2.0 * len(sentence_terms.intersection(key_terms)) + 3.0 * sum(1 for t in sentence_terms if '=' in t)
            score += sum(1.5 for term in prompt.lower().split() if term in sentence_lower)
            score -= sum(2.0 for term in ['galvanometer', 'torque', 'ampère', 'sensitivity', 'newton'] if term in sentence_lower and term not in prompt.lower())
            if score > 5.0:
                definition_sentences.append((cleaned, score))
        
        definition_sentences.sort(key=lambda x: x[1], reverse=True)
        definition_sentences = [s[0] for s in definition_sentences][:3]
        
        if definition_sentences:
            response += " ".join(definition_sentences)
        else:
            response += self._clean_sentence(content[:400])
        response += ("..." if len(content) > 400 else "")
        
        math_patterns = re.findall(r'\b[\w]\s*=\s*[\w\d\s/+-∑∮]+', content)
        math_patterns = [eq.strip() for eq in math_patterns if any(term in eq.lower() for term in key_terms)][:2]
        if math_patterns:
            response += "\n**Related Equations:**\n"
            for eq in math_patterns:
                response += f"- \\({eq}\\)\n"
        
        return response