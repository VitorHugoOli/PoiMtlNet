#!/usr/bin/env python3
"""Gera SPEECH.md e SPEECH.tex a partir do SLIDES.md (a fonte da fala) e do PDF construido.

Desenho: NAO e' uma transcricao. A fala media tem ~158 palavras, o que e' ilegivel de
relance. Cada slide vira um CARTAO com quatro camadas, em ordem de urgencia:

    1. ABRE   -- a primeira oracao, para pegar o fio sem ler
    2. DIZER  -- os trechos que o SLIDES.md marca em negrito: sao as superficies de lei
                 e os numeros que nao podem sair errado
    3. NUNCA  -- o campo 'Nunca dizer' do bloco, condensado
    4. a fala completa, em corpo menor, para consulta

Nada e' inventado: tudo sai do bloco correspondente do SLIDES.md.
"""
import re, subprocess, sys, unicodedata

MD = 'SLIDES.md'
TEX = 'slides/main.tex'
PDF = 'slides/main.pdf'


def limpa(s):
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_blocos():
    t = open(MD, encoding='utf-8').read()
    out = []
    for m in re.finditer(r'\n### (S\d+) · (.+?)\n(.*?)(?=\n### |\n# |\Z)', t, re.S):
        code, title, body = m.group(1), m.group(2).strip(), m.group(3)
        if code.startswith('SB'):
            continue
        def campo(nome, ate=r'\n- \*\*'):
            mm = re.search(r'\*\*' + nome + r':?\*\*(.*?)(?=' + ate + r'|\Z)', body, re.S)
            return limpa(mm.group(1)) if mm else ''
        tempo = ''
        mt = re.search(r'\*\*Tempo:\*\*\s*([^\n]+)', body)
        if mt:
            tempo = limpa(mt.group(1))
        out.append(dict(code=code, title=title,
                        tempo=tempo,
                        fala=campo(r'Fala \(PT\)'),
                        nunca=campo(r'Nunca dizer'),
                        ledger=campo(r'LEDGER')))
    return out


def paginas():
    """titulo do frame -> lista de (pagina_pdf, numero_impresso), na ordem do PDF."""
    pgs = subprocess.run(['pdftotext', PDF, '-'], capture_output=True, text=True).stdout.split('\f')
    info = []
    for i, p in enumerate(pgs, 1):
        nums = [int(x) for x in re.findall(r'(?m)^\s*(\d{1,2})\s*$', p)]
        info.append((i, nums[-1] if nums else None, p))
    return info


# titulos que diferem entre o SLIDES.md e o frame construido
ALIAS = {
    'Protocol, step 1 of 4: the unit of data':   '1 · the unit of data',
    'Protocol, step 2 of 4: what is measured':   '2 · what is measured',
    'Protocol, step 3 of 4: what is compared':   '3 · what is compared',
    'Protocol, step 4 of 4: how it is decided':  '4 · how it is decided',
    'Multitask Learning for POI Classification and Prediction Tasks': 'Defesa de Dissertação de Mestrado',
}


def acha_pagina(title, info, desde):
    title = ALIAS.get(title, title)
    """primeira pagina >= desde cujo texto contem o titulo (normalizado)."""
    alvo = limpa(re.sub(r'[^\w\s]', ' ', title)).lower()[:34]
    for i, num, p in info:
        if i < desde:
            continue
        if alvo and alvo in limpa(re.sub(r'[^\w\s]', ' ', p)).lower():
            return i, num
    return None, None


def segundos(tempo):
    m = re.search(r'([\d,\.]+)\s*(s|min)', tempo)
    if not m:
        return 0
    v = float(m.group(1).replace(',', '.'))
    return int(v * 60 if m.group(2) == 'min' else v)


def mmss(s):
    return '%d:%02d' % (s // 60, s % 60)


def negritos(fala):
    """os trechos que o SLIDES.md marca em negrito na fala = as superficies que nao podem sair errado."""
    b = re.findall(r'\*\*(.+?)\*\*', fala)
    vistos, out = set(), []
    for x in b:
        x = limpa(x).strip(' .,;:')
        k = x.lower()
        if len(x) > 3 and k not in vistos:
            vistos.add(k)
            out.append(x)
    return out


def abre(fala):
    f = re.sub(r'^["“]', '', fala.strip())
    f = re.sub(r'\*\*', '', f)
    m = re.match(r'(.{0,180}?[\.\?!])(\s|$)', f)
    out = limpa(m.group(1)) if m else limpa(f[:170]) + '…'
    if len(out) < 45:                      # parou cedo demais: pega a oracao seguinte tambem
        m2 = re.match(r'(.{0,220}?[\.\?!])(\s|$)', f[len(out):].strip())
        if m2:
            out = out + ' ' + limpa(m2.group(1))
    return out


def nunca_curto(n):
    n = re.sub(r'⚠.*$', '', n)
    n = re.sub(r'\*\*|\*|`', '', n)
    partes = [limpa(x) for x in re.split(r'(?<=[\.;])\s+', n) if len(limpa(x)) > 3]
    return partes[:4]


def main():
    blocos = parse_blocos()
    info = paginas()
    cursor, acum = 1, 0
    for b in blocos:
        pg, num = acha_pagina(b['title'], info, cursor)
        if pg:
            cursor = pg + 1
        b['pdf'], b['num'] = pg, num
        b['seg'] = segundos(b['tempo'])
        acum += b['seg']
        b['acum'] = acum
        b['abre'] = abre(b['fala'])
        b['dizer'] = negritos(b['fala'])
        b['nunca_l'] = nunca_curto(b['nunca'])
    print("blocos: %d | mapeados a pagina: %d | tempo total declarado: %s"
          % (len(blocos), sum(1 for b in blocos if b['pdf']), mmss(acum)))
    for b in blocos[:6]:
        print("  %-5s pdf p%-4s impresso %-5s %-46s %ss" % (b['code'], b['pdf'], b['num'], b['title'][:46], b['seg']))
    import json
    json.dump(blocos, open('/tmp/_speech_blocos.json', 'w'), ensure_ascii=False)
    return blocos


if __name__ == '__main__':
    main()
