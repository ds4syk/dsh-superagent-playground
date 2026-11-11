"""Small CJK typesetter on top of PyMuPDF: measure, wrap, justify, draw."""
import pymupdf as fitz, re, os, math

FONT_FILES={
 "regular":"/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
 "bold":"/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
}
_fonts={}
def font(kind="regular"):
    if kind not in _fonts:
        _fonts[kind]=fitz.Font(fontfile=FONT_FILES[kind])
    return _fonts[kind]

CJK=re.compile(r'[\u2e80-\u9fff\uff00-\uffef\u3000-\u303f]')

def tokenize(text, math_map):
    """split into tokens: ('c', char) for CJK, ('w', word) for latin runs, ('m', mathitem), ('s', ' ') for spaces"""
    toks=[]
    i=0
    for piece in re.split(r'(⟦\d+⟧)', text):
        if not piece: continue
        m=re.fullmatch(r'⟦(\d+)⟧', piece)
        if m:
            item=math_map.get(int(m.group(1)))
            if item is not None: toks.append(('m', item))
            continue
        for ch in piece:
            if ch.isspace(): toks.append(('s',' '))
            elif CJK.match(ch): toks.append(('c',ch))
            else: toks.append(('l',ch))
    # merge consecutive latin chars into words (keep punctuation attached)
    out=[]; buf=""
    for t in toks:
        if t[0]=='l': buf+=t[1]
        else:
            if buf: out.append(('w',buf)); buf=""
            out.append(t)
    if buf: out.append(('w',buf))
    return out

def tok_width(tok, size, kind="regular"):
    f=font(kind)
    if tok[0]=='m':
        it=tok[1]; return it["w"]
    if tok[0]=='s': return size*0.30
    return f.text_length(tok[1], fontsize=size)

def layout(tokens, width, size, kind="regular", first_indent=0.0):
    f=font(kind)
    lines=[]; cur=[]; curw=0.0; avail=width-first_indent
    breakable_after=lambda tok: tok[0] in ('c','s') or (tok[0]=='w' and re.search(r'[-/）)；，。、,;:]\Z', tok[1]))
    i=0
    while i < len(tokens):
        tok=tokens[i]
        if tok[0]=='m' and tok[1].get("w",0)>0.55*width and cur:
            lines.append({"toks":cur,"w":curw,"indent":first_indent if not lines else 0.0})
            cur=[]; curw=0.0; avail=width
        # keep latin word + following punctuation together
        w=tok_width(tok,size,kind)
        if curw + w <= avail + 0.01 or not cur:
            cur.append(tok); curw+=w; i+=1
            continue
        # try to move a trailing latin word to next line
        cut=len(cur)
        while cut>0 and cur[cut-1][0]=='w':
            cut-=1
        if cut==0 or cut==len(cur):
            lines.append({"toks":cur,"w":curw,"indent":first_indent if not lines else 0.0}); cur=[]; curw=0.0; avail=width
            continue
        moved=cur[cut:]; lines.append({"toks":cur[:cut],"w":sum(tok_width(t,size,kind) for t in cur[:cut]),
                                       "indent":first_indent if not lines else 0.0})
        cur=moved; curw=sum(tok_width(t,size,kind) for t in moved); avail=width
    if cur: lines.append({"toks":cur,"w":curw,"indent":first_indent if not lines else 0.0})
    return lines

def line_height(ln, leading):
    mh=0.0
    for t in ln["toks"]:
        if t[0]=='m':
            mh=max(mh, t[1].get("h",0)+1.5)
    return max(leading, mh)

def natural_height(tokens, width, size, leading, kind="regular"):
    ls=layout(tokens,width,size,kind)
    h=0.0
    for i,ln in enumerate(ls):
        h += line_height(ln,leading) if i>0 else max(size, line_height(ln,leading))
    return h, ls

def draw(page, tokens, rect, size, leading, align="justify", kind="regular",
         color=(0,0,0), last_line_left=True, baseline0=None):
    """Draw tokens inside rect (x0,y0,x1,y1). Returns used height."""
    x0,y0,x1,y1=rect
    width=x1-x0
    ls=layout(tokens,width,size,kind)
    f=font(kind)
    base = y0 + size*0.86 if baseline0 is None else baseline0
    tw=fitz.TextWriter(page.rect, color=color)
    imgs=[]
    y=base
    for li,ln in enumerate(ls):
        if li>0: y += line_height(ls[li-1], leading)
        extra=0.0
        if align=="justify" and li<len(ls)-1:
            gaps=sum(1 for t in ln["toks"][:-1] if t[0] in ('c','s'))
            spare=width-ln["w"]
            if gaps>=5 and spare>0 and spare/gaps<=0.22*size: extra=spare/gaps
            elif gaps>=5 and spare>0 and spare<=0.06*width: extra=spare/gaps
            else: extra=0.0
        x=x0+ln["indent"]
        for t in ln["toks"]:
            if t[0]=='m':
                it=t[1]
                top=it.get("top",it["h"]); bot=it.get("bot",0.0)
                if not (top==top and bot==bot): top, bot = it["h"], 0.0
                w=max(0.6,it["w"])
                r=fitz.Rect(x, y-top, x+w, y+bot)
                if r.width>0.6 and r.height>0.6: imgs.append((r, it["file"]))
                x+=it["w"]
            else:
                s=' ' if t[0]=='s' else t[1]
                if t[0]=='w': tw.append((x,y), s, font=f, fontsize=size)
                else: tw.append((x,y), s, font=f, fontsize=size)
                x+=tok_width(t,size,kind)
            if t[0] in ('c','s'): x+=extra
    tw.write_text(page, color=color)
    for r,fn in imgs:
        page.insert_image(r, filename=os.path.join("work",fn) if not os.path.isabs(fn) else fn, overlay=True)
    used=sum(line_height(l,leading) for l in ls[:-1]) + (ls[-1]["w"]*0+leading if False else max(size, line_height(ls[-1],leading)))
    return used, len(ls)
