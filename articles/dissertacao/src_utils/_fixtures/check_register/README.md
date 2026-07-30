# _fixtures/check_register

Two tiny hand-written trees proving `check_register.py` fires on the register defects and stays
silent on their American equivalents. Not copies of the real tree (failure class T6).

`dirty/src/chapters/fixture_target.tex` carries one hit of each mechanical class the gate must
catch: the British `need`+gerund construction ("needs saying plainly", the author's own instance
from PENDENCIAS 2.22 point 8), three British spellings (`neighbour`, `behaviour`, `centre`), and a
Class B delayed-subject shape ("Several differences between the two configurations emerge").
`clean/` is the same tree with each rewritten in American English and with the subject named first.

Two shapes of the fixture are load-bearing and will look arbitrary otherwise:

- **Twenty filler chapters.** The gate refuses to report on a scope below 20 `.tex` files, because
  two other prose gates on this project silently swept almost nothing after a file move. A fixture
  under the floor would exit 2 on both sides.
- **Stubs at the two real OPEN_REGISTER paths** (`chapters/apx_f_cosine.tex` and
  `chapters/3_cbic/conclusion.tex`), each holding the needle its register entry names. The open
  register is self-retiring: an entry whose needle is absent makes the gate FAIL and demand the
  entry's deletion. Without these stubs the clean side would fail, and it would be the gate working
  correctly rather than a fixture bug.

`references.bib` carries `towards`, `Behaviour` and `Colour` inside `title` and `journal`, which are
attributes of record under AGENT_GUARDRAILS §1 R2. **Both sides must stay silent on them.** That is
the point of including them: the real bibliography's only British form is inside a published title
(Xu2023, `towards`, confirmed at Crossref 10.1145/3582553), and a gate that "corrected" it would
corrupt a citation attribute.
