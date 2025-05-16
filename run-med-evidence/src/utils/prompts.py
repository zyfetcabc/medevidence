from utils.CONSTS import ANSWER_CLASSES

_REC_TABLE = """
    **REC TABLE **: Levels of Evidence (from strongest [1a] to lowest [5]).  

    | Grade of Recommendation | Level of Evidence | Type of Study |  
    |-------------------------|-------------------|---------------------------------------------------------------------|  
    | A | 1a | Systematic review and meta-analysis of (homogeneous) randomized controlled trials |  
    | A | 1b | Individual randomized controlled trials (with narrow confidence intervals) |  
    | B | 2a | Systematic review of (homogeneous) cohort studies of 'exposed' and 'unexposed' subjects |  
    | B | 2b | Individual cohort study / low-quality randomized control studies |  
    | B | 3a | Systematic review of (homogeneous) case-control studies |  
    | B | 3b | Individual case-control studies |  
    | C | 4 | Case series, low-quality cohort or case-control studies, or case reports |  
    | D | 5 | Expert opinions based on non-systematic reviews of results or mechanistic studies |" 
"""

_RELEVANCE_CHECK = 'If the article does not contain relevant information to the QUESTION, summarize the article ONLY with the phrase "N/A". Do not include anything else in your article summary.'

BASIC_PROMPTS = dict(
    get_info = """As a subject expert, (1) summarize the evidence provided by a given ARTICLE as it pertains to a given QUESTION and (2) provide a possible answer.
    {REL_CHECK}
    If the provided article contains relevant information, return a list including the following items:
    - **Summary**: A concise but comprehensive summary based on the previously specified information, with a focus on the main findings.
    - **Possible Answer**: A concise feasible answer given the evidence.

    Think step by step.
    {text}
    """,

    get_info_values = "**QUESTION**: {question}\n**ARTICLE TITLE**: {title}\n**ARTICLE CONTENT**:\n{context}",

    final_answer = """Given the ARTICLE SUMMARIES. Provide a concise and precise answer to the provided QUESTION.

    After you think, return your answer with the following format:
    - **Rationale**: Your rationale
    - **Full Answer**:  A precise answer, citing each fact with the Article ID in brackets (e.g. [2]).
    - **Answer**: A final classification exactly matching one of the following options: {CLASSES}

    Think step by step.
    {text}
    """,
    ### Str add values to final answer prompt ###
    final_answer_values = "\n\n**QUESTION**: '{question}'\n **ARTICLE SUMMARIES**: {context}"
)

EVIDENCE_AWARE_PROMPTS = dict(
    ### Prompt to summarize evidence ###
    get_info = """As a subject expert, (1) summarize the evidence provided by a given ARTICLE as it pertains to a given QUESTION and (2) provide a possible answer.
    {REL_CHECK}
    Otherwise, if the provided article contains relevant information, you must return a list including the following items:

    - **Study Design**: Type of study, level of evidence, and grade of recommendation according to the levels of evidence REC TABLE (provided Below).  
    - **Study Population**: Study size and patient population.
    - **Summary**: A concise but comprehensive summary based on the previously specified information, with a focus on the main findings.
    - **Possible Answer**: A concise feasible answer given the evidence.
    """
    +_REC_TABLE+
    """
    Think step by step.
    {text}
    """,

    ### Prompt to generate  final answer (given all evidence) ###
    final_answer = BASIC_PROMPTS['final_answer'],
    get_info_values=BASIC_PROMPTS['get_info_values'],
    final_answer_values=BASIC_PROMPTS['final_answer_values']
)

QUALITY_PRIORITIZING_PROMPTS = dict(
    get_info=EVIDENCE_AWARE_PROMPTS['get_info'],
    get_info_values=BASIC_PROMPTS['get_info_values'],
    final_answer_values=BASIC_PROMPTS['final_answer_values'],
    ### Prompt to generate  final answer (given all evidence) ###
    final_answer="""Given the ARTICLE SUMMARIES. Provide a concise and precise answer to the provided QUESTION.
    To this end, prioritize summaries with the highest levels of evidence, from '1a' the highest to '5' the lowest (REC TABLE provides the full hierarchy of evidence) and publishing date (as in a newer article is better if and only if it has a higher level of evidence).
    Lastly, if articles have the same level of evidence, prioritize the one with the largest population and lowest risk of bias.
    """
    +_REC_TABLE+
    """
    After you think, return your answer with the following format:
    - **Rationale**: Your rationale
    - **Full Answer**:  A precise answer, citing each fact with the Article ID in brackets (e.g. [2]).
    - **Answer**: A final classification exactly matching one of the following options: {CLASSES}

    Think step by step.
    {text}
    """
)

_COCHRANE_CONTEXT = "You are the author of a Cochrane Collaboration systematic review, leveraging statistical analysis and assessing risks of bias " \
                  + "in order to rigorously assess the effectiveness of medical interventions. As part of your review process, perform the following task:\n"
COCHRANE_CONTEXT_PROMPTS = dict(
    get_info=_COCHRANE_CONTEXT + BASIC_PROMPTS['get_info'],
    get_info_values=BASIC_PROMPTS['get_info_values'],
    final_answer=_COCHRANE_CONTEXT + BASIC_PROMPTS['final_answer'],
    final_answer_values=BASIC_PROMPTS['final_answer_values']
)

COCHRANE_PLUS_PROMPTS = dict(
    get_info=_COCHRANE_CONTEXT + EVIDENCE_AWARE_PROMPTS['get_info'],
    get_info_values=BASIC_PROMPTS['get_info_values'],
    final_answer=_COCHRANE_CONTEXT + BASIC_PROMPTS['final_answer'],
    final_answer_values=BASIC_PROMPTS['final_answer_values']
)

SINGLE_SOURCE_PROMPTS = dict(
    get_info=BASIC_PROMPTS['get_info'],
    final_answer_values=BASIC_PROMPTS['final_answer_values'],
    final_answer = """Given the ARTICLE SUMMARIES, provide a concise and precise answer to the provided QUESTION.
    When determining your Answer class, note that differences should only be considered valid if they are statistically significant.
    If the treatment, control, and outcome from the question are not explicitly evaluated in the given ARTICLE SUMMARIES, the correct Answer should be Insufficient Data.

    After you think, return your answer with the following format:
    - **Rationale**: Your rationale
    - **Full Answer**:  A precise answer, citing each fact with the Article ID in brackets (e.g. [2]).
    - **Answer**: A final classification exactly matching one of the following options: {CLASSES}

    Think step by step.
    {text}
    """
)

############

PROMPT_SETS = {
    'basic': BASIC_PROMPTS,
    'quality_aware': EVIDENCE_AWARE_PROMPTS,
    'quality_priority': QUALITY_PRIORITIZING_PROMPTS,
    'cochrane_aware': COCHRANE_CONTEXT_PROMPTS,
    'cochrane_plus': COCHRANE_PLUS_PROMPTS,
    'source_level_agreement': SINGLE_SOURCE_PROMPTS
}

def load_prompts(prompt_set, check_relevance=True):
    assert prompt_set in PROMPT_SETS, f"{prompt_set} is not a recognized prompt set"
    prompts = PROMPT_SETS[prompt_set]
    prompts['get_info'] = prompts['get_info'].replace(
        "{REL_CHECK}", (_RELEVANCE_CHECK if check_relevance else ""))
    prompts['final_answer'] = prompts['final_answer'].replace(
        "{CLASSES}", ', '.join(ANSWER_CLASSES))
    return prompts