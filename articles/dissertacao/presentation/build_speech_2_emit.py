#!/usr/bin/env python3
"""Emite SPEECH.md e SPEECH.tex a partir dos blocos extraidos por gen_speech.py."""
import json, re, sys

B = json.load(open('/tmp/_speech_blocos.json'))

SECOES = [
    ('S1',  'ABERTURA — a pergunta e o escopo', 5),
    ('S8',  'FUNDAMENTOS — dito uma vez', 6),
    ('S17', 'MTLnet — Cap. 3 (CBIC)', 5.5),
    ('S26', 'ST-MTLNet — Cap. 4 (CoUrb)', 6),
    ('S33', 'Check2HGI — Cap. 5 (MobiWac)', 20),
    ('S50', 'CONCLUSÃO — a resposta condicional', 5.5),
]


def mmss(s):
    return '%d:%02d' % (s // 60, s % 60)


def numeros_da_fala(f):
    """So' numeros que sao RESULTADO: com separador decimal, ou seguidos de 'por cento'/'pp'.
    Inteiro nu (Secao 5, Capitulo 3) e' referencia, nao resultado, e polui.
    Recorta por TOKEN, nunca por caractere, para nao partir palavra ao meio."""
    f = re.sub(r'\*\*', '', f)
    toks = f.split()
    alvo = re.compile(r'^\(?\d+[,\.]\d+\)?[,;:\.]?$')
    pct = re.compile(r'^\d+$')
    out, vistos, usados = [], set(), set()
    for i, t in enumerate(toks):
        eh = bool(alvo.match(t)) or (pct.match(t) and i + 1 < len(toks)
                                     and toks[i + 1].lower().startswith(('por', 'pp')))
        if not eh or i in usados:
            continue
        ini, fim = max(0, i - 4), min(len(toks), i + 4)
        for k in range(ini, fim):
            usados.add(k)
        frag = ' '.join(toks[ini:fim]).strip(' ,;:.')
        k = frag.lower()
        if k not in vistos and len(frag) > 6:
            vistos.add(k); out.append(frag)
    return out[:4]


def limpa(s):
    return re.sub(r'\s+', ' ', s).strip()


def dizer_util(lst):
    """so' o que nao pode sair errado: superficie longa, ou trecho com numero."""
    out = []
    for x in lst:
        if len(x) >= 14 or re.search(r'\d', x):
            out.append(x)
    if not out:
        out = [x for x in lst if len(x) >= 6]
    return out[:6]


def fala_limpa(f):
    f = re.sub(r'^["“]|["”]$', '', f.strip())
    return f


def sec_de(code):
    n = int(code[1:])
    atual = SECOES[0]
    for c, nome, mins in SECOES:
        if int(c[1:]) <= n:
            atual = (c, nome, mins)
    return atual[1]


# ─────────────────────────── MARKDOWN ───────────────────────────
def emit_md():
    L = []
    tot = B[-1]['acum']
    L.append('# SPEECH.md — o que dizer, slide a slide\n')
    L.append('> **Defesa · sexta, 28/08/2026, 10:00 · remota (Google Meet).** Gerado do '
             '`SLIDES.md`, que é a fonte da fala. **Nada aqui é novo**: é o mesmo texto, '
             'reorganizado para ser lido de relance em vez de lido inteiro.\n')
    L.append('> **Como usar.** Cada cartão tem quatro camadas, em ordem de urgência: '
             '**ABRE** (a primeira oração, para pegar o fio sem ler), **DIZER** (as superfícies '
             'de lei e os números que não podem sair errado), **NUNCA** (o que anula o slide se '
             'escapar), e a fala completa embaixo, para consulta.\n')
    L.append('> ⚠ **O relógio.** Os tempos abaixo são os do plano e somam **%s**. A fala escrita '
             'tem 8.840 palavras, que a 140 palavras/minuto dão **~63 min** — contra o teto de '
             '**50 min** do Art. 23. Os dois números não fecham, e o ensaio é que decide qual '
             'vale. **Cronometre o fim de cada seção.**\n' % mmss(tot))
    L.append('\n---\n')

    # sumario por secao
    L.append('## Marcas de tempo — leve estas seis\n')
    L.append('| seção | slides | fim previsto | **seu tempo real** |')
    L.append('|---|---|---:|---|')
    for c, nome, mins in SECOES:
        fim = [b for b in B if int(b['code'][1:]) >= int(c[1:])]
        idx = int(c[1:])
        ate = [b for b in B if int(b['code'][1:]) < idx + 100]
        último = max([b for b in B if sec_de(b['code']) == nome], key=lambda x: int(x['code'][1:]))
        prim = min([b for b in B if sec_de(b['code']) == nome], key=lambda x: int(x['code'][1:]))
        L.append('| **%s** | %s–%s | %s | ____________ |' % (nome, prim['code'], último['code'], mmss(último['acum'])))
    L.append('')
    L.append('\n---\n')

    sec_atual = None
    for b in B:
        s = sec_de(b['code'])
        if s != sec_atual:
            sec_atual = s
            L.append('\n\n# %s\n' % s)
        loc = []
        if b['num']:
            loc.append('slide **%s**' % b['num'])
        if b['pdf']:
            loc.append('PDF p.%s' % b['pdf'])
        L.append('\n## %s · %s' % (b['code'], b['title']))
        L.append('`%s` · **%s** · fim previsto **%s**\n' % (' · '.join(loc), b['tempo'], mmss(b['acum'])))
        if b['abre']:
            L.append('> ### ▶ %s' % b['abre'])
        d = dizer_util(b['dizer'])
        if d:
            L.append('\n**● DIZER EXATO**')
            for x in d:
                L.append('- %s' % x)
        elif b['ledger']:
            L.append('\n**● COBRE** — %s' % re.sub(r'\*\*|`', '', b['ledger'])[:230])
        nums = numeros_da_fala(b['fala'])
        if nums:
            L.append('\n**# NÚMEROS** — %s' % ' · '.join(nums))
        if b['nunca_l']:
            L.append('\n**✕ NUNCA** — %s' % ' · '.join(b['nunca_l']))
        L.append('\n<sub>%s</sub>\n' % fala_limpa(b['fala']))
        L.append('---')
    open('SPEECH.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('SPEECH.md escrito (%d cartões)' % len(B))


# ─────────────────────────── LATEX ───────────────────────────
def tex_esc(s):
    for a, c in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')):
        s = s.replace(a, c)
    s = s.replace('“', '``').replace('”', "''").replace('—', '---').replace('·', r'$\cdot$')
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    return s


def emit_tex():
    tot = B[-1]['acum']
    P = [r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[margin=15mm,top=13mm,bottom=13mm]{geometry}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[brazilian]{babel}
\tcbuselibrary{breakable}
\definecolor{teal}{HTML}{1C7A72}
\definecolor{crim}{HTML}{A83455}
\definecolor{ink}{HTML}{131C1A}
\definecolor{soft}{HTML}{5B6B68}
\definecolor{band}{HTML}{EAF2F0}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\parindent}{0pt}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small\color{soft}SPEECH — defesa 28/08/2026}
\fancyhead[R]{\small\color{soft}\thepage}
\begin{document}
\begin{center}
{\Huge\bfseries\color{teal} O que dizer, slide a slide}\\[3mm]
{\large Defesa de Mestrado \textbf{28/08/2026, 10:00} --- remota (Google Meet)}
\end{center}
\vspace{3mm}
\begin{tcolorbox}[colback=band,colframe=teal,boxrule=0.8pt,arc=2pt]
\textbf{Como usar.} Cada cartão tem quatro camadas, em ordem de urgência:
\textbf{\textcolor{teal}{ABRE}} (a primeira oração, para pegar o fio sem ler),
\textbf{\textcolor{teal}{DIZER}} (as superfícies de lei e os números que não podem sair errado),
\textbf{\textcolor{crim}{NUNCA}} (o que anula o slide se escapar), e a fala completa embaixo.
\textbf{Nada aqui é novo} --- é o texto do \texttt{SLIDES.md}, reorganizado para ser lido de relance.
\smallskip

\textbf{\textcolor{crim}{O relógio.}} Os tempos abaixo são os do plano e somam ''' + mmss(tot) + r'''.
A fala escrita tem \textbf{8.840 palavras}, que a 140 palavras/minuto dão \textbf{$\sim$63 min},
contra o teto de \textbf{50 min} do Art.~23. Os dois números não fecham.
\textbf{Cronometre o fim de cada seção} e anote na tabela abaixo.
\end{tcolorbox}
\vspace{4mm}

{\large\bfseries\color{teal} Marcas de tempo --- leve estas seis}
\vspace{2mm}

\begin{tabular}{@{}p{62mm} p{22mm} r p{34mm}@{}}
\hline
\textbf{seção} & \textbf{slides} & \textbf{previsto} & \textbf{seu tempo real} \\
\hline
''']
    for c, nome, mins in SECOES:
        grp = [b for b in B if sec_de(b['code']) == nome]
        if not grp:
            continue
        prim = min(grp, key=lambda x: int(x['code'][1:]))
        ult = max(grp, key=lambda x: int(x['code'][1:]))
        P.append(r'\textbf{%s} & %s--%s & %s & \rule{32mm}{0.4pt} \\' %
                 (tex_esc(nome), prim['code'], ult['code'], mmss(ult['acum'])))
    P.append(r'''\hline
\end{tabular}
\clearpage
''')

    sec_atual = None
    for b in B:
        s = sec_de(b['code'])
        if s != sec_atual:
            sec_atual = s
            P.append(r'\vspace{2mm}{\LARGE\bfseries\color{teal} %s}\par\vspace{2mm}' % tex_esc(s))
        loc = []
        if b['num']:
            loc.append(r'slide \textbf{%s}' % b['num'])
        if b['pdf']:
            loc.append('PDF p.%s' % b['pdf'])
        P.append(r'\begin{tcolorbox}[breakable,colback=white,colframe=teal!55,boxrule=0.6pt,arc=2pt,'
                 r'left=3mm,right=3mm,top=2mm,bottom=2mm]')
        P.append(r'{\small\color{soft}\textbf{%s} $\cdot$ %s $\cdot$ %s $\cdot$ fim previsto \textbf{%s}}\par\vspace{1mm}'
                 % (b['code'], ' $\\cdot$ '.join(loc), tex_esc(b['tempo']), mmss(b['acum'])))
        P.append(r'{\large\bfseries\color{ink} %s}\par\vspace{2mm}' % tex_esc(b['title']))
        if b['abre']:
            P.append(r'{\color{teal}\textbf{\textrightarrow}}\ \textbf{\itshape %s}\par\vspace{1.5mm}' % tex_esc(b['abre']))
        d = dizer_util(b['dizer'])
        if d:
            P.append(r'{\small\color{teal}\textbf{DIZER EXATO}}')
            P.append(r'\begin{itemize}[leftmargin=6mm,itemsep=0.4mm,topsep=0.8mm]')
            for x in d:
                P.append(r'\item %s' % tex_esc(x))
            P.append(r'\end{itemize}')
        elif b['ledger']:
            P.append(r'{\small\color{teal}\textbf{COBRE} --- %s}\par\vspace{1mm}'
                     % tex_esc(re.sub(r'\*\*|`', '', b['ledger'])[:230]))
        nums = numeros_da_fala(b['fala'])
        if nums:
            P.append(r'\vspace{0.5mm}{\small\textbf{N\'UMEROS} --- %s}\par' % ' $\\cdot$ '.join(tex_esc(x) for x in nums))
        if b['nunca_l']:
            P.append(r'\vspace{0.5mm}{\small\color{crim}\textbf{NUNCA} --- %s}\par' %
                     ' $\\cdot$ '.join(tex_esc(x) for x in b['nunca_l']))
        P.append(r'\vspace{1.5mm}{\footnotesize\color{soft} %s\par}' % tex_esc(fala_limpa(b['fala'])))
        P.append(r'\end{tcolorbox}\vspace{2mm}')
    P.append(r'\end{document}')
    open('SPEECH.tex', 'w', encoding='utf-8').write('\n'.join(P) + '\n')
    print('SPEECH.tex escrito')


emit_md()
emit_tex()
