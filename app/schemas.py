from pydantic import BaseModel, Field


class SourceResponse(BaseModel):
    sourceId: str
    filename: str
    chunkCount: int


class Problem(BaseModel):
    question: str
    choices: list[str] = Field(min_length=4, max_length=4)
    answerIndex: int = Field(ge=0, le=3)


class ProblemSet(BaseModel):
    count: int
    problems: list[Problem]


class ProblemRequest(BaseModel):
    sourceId: str


class GradeRequest(BaseModel):
    studentId: str
    answerIndexes: list[int]


class GradeResponse(BaseModel):
    studentId: str
    score: int
    total: int
    detectedAnswers: list[int]
    correctAnswers: list[int]
