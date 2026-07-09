"""Manim scenes for The Tell. Shared palette = the paper's colors.
Render a still:  python -m manim -s -qh --format=png --transparent scripts/scenes.py <Scene> -o <name>
"""
from manim import *

Text.set_default(font="Helvetica Neue")

INK  = ManimColor("#16324F")
ACC  = ManimColor("#B5172F")
GRN  = ManimColor("#1B7837")
AMB  = ManimColor("#C77D11")
GREY = ManimColor("#7A7A7A")
LINK = ManimColor("#E8EEF4")   # light ink
LACC = ManimColor("#F6E2E5")   # light accent
LGRN = ManimColor("#E2F0E6")   # light green


class LayerStack(Scene):
    def construct(self):
        n = 7
        layers = VGroup(*[
            RoundedRectangle(width=4.4, height=0.52, corner_radius=0.08,
                             stroke_color=INK, stroke_width=2, fill_color=LINK, fill_opacity=1)
            for _ in range(n)])
        layers.arrange(UP, buff=0.13)
        mid = layers[n // 2]
        mid.set_stroke(ACC, width=4); mid.set_fill(LACC, 1)

        reply = RoundedRectangle(width=5.4, height=0.72, corner_radius=0.1,
                                 stroke_color=GRN, stroke_width=2.5, fill_color=LGRN, fill_opacity=1)
        rtext = Text('"Your refund is processed, anything else?"', font_size=22, color=INK)
        rtext.scale_to_fit_width(reply.width - 0.35).move_to(reply)
        reply_g = VGroup(reply, rtext).next_to(layers, UP, buff=0.55)
        arr_up = Arrow(layers.get_top(), reply.get_bottom(), color=INK, buff=0.05, stroke_width=5,
                       max_tip_length_to_length_ratio=0.4)

        req = Text("the request goes in here", font_size=22, color=GREY).next_to(layers, DOWN, buff=0.32)

        probe = RoundedRectangle(width=2.9, height=1.0, corner_radius=0.1,
                                 stroke_color=ACC, stroke_width=2.5, fill_color=LACC, fill_opacity=1)
        ptext = VGroup(Text("we read here", font_size=24, color=ACC, weight=BOLD),
                       Text("one middle layer", font_size=18, color=INK)).arrange(DOWN, buff=0.09).move_to(probe)
        probe_g = VGroup(probe, ptext).next_to(mid, RIGHT, buff=1.7)
        arr_probe = Arrow(probe.get_left(), mid.get_right(), color=ACC, buff=0.12, stroke_width=5)

        left = Paragraph("the underneath:", "the leaning,", "layer by layer",
                         font_size=19, color=GREY, line_spacing=0.7, alignment="left").next_to(layers, LEFT, buff=0.8)
        surf = Paragraph("the surface:", "what the agent says",
                         font_size=17, color=GREY, slant=ITALIC, line_spacing=0.7, alignment="left").next_to(reply_g, RIGHT, buff=0.45)

        whole = VGroup(layers, reply_g, arr_up, req, probe_g, arr_probe, left, surf)
        whole.scale(0.82).move_to(ORIGIN)
        self.add(whole)


class LayerStackDark(Scene):
    def construct(self):
        self.camera.background_color = ManimColor("#0E1117")
        BOXF = ManimColor("#18232F"); BOXS = ManimColor("#6FA8DC")
        MIDS = ManimColor("#FF6B7D"); MIDF = ManimColor("#3A1B22")
        GRNS = ManimColor("#57D9A3"); GRNF = ManimColor("#153029")
        LT = ManimColor("#E8EEF4"); DIM = ManimColor("#9BA7B4")
        n = 7
        layers = VGroup(*[
            RoundedRectangle(width=4.4, height=0.52, corner_radius=0.08,
                             stroke_color=BOXS, stroke_width=2, fill_color=BOXF, fill_opacity=1)
            for _ in range(n)]).arrange(UP, buff=0.13)
        mid = layers[n // 2]; mid.set_stroke(MIDS, width=4.5); mid.set_fill(MIDF, 1)
        glow = RoundedRectangle(width=4.7, height=0.8, corner_radius=0.1,
                                stroke_color=MIDS, stroke_width=1, fill_opacity=0).move_to(mid).set_opacity(0.25)

        reply = RoundedRectangle(width=5.4, height=0.72, corner_radius=0.1,
                                 stroke_color=GRNS, stroke_width=2.5, fill_color=GRNF, fill_opacity=1)
        rtext = Text('"Your refund is processed, anything else?"', font_size=22, color=LT)
        rtext.scale_to_fit_width(reply.width - 0.35).move_to(reply)
        reply_g = VGroup(reply, rtext).next_to(layers, UP, buff=0.55)
        arr_up = Arrow(layers.get_top(), reply.get_bottom(), color=BOXS, buff=0.05, stroke_width=5,
                       max_tip_length_to_length_ratio=0.4)
        req = Text("the request goes in here", font_size=22, color=DIM).next_to(layers, DOWN, buff=0.32)

        probe = RoundedRectangle(width=2.9, height=1.0, corner_radius=0.1,
                                 stroke_color=MIDS, stroke_width=2.5, fill_color=MIDF, fill_opacity=1)
        ptext = VGroup(Text("we read here", font_size=24, color=MIDS, weight=BOLD),
                       Text("one middle layer", font_size=18, color=LT)).arrange(DOWN, buff=0.09).move_to(probe)
        probe_g = VGroup(probe, ptext).next_to(mid, RIGHT, buff=1.7)
        arr_probe = Arrow(probe.get_left(), mid.get_right(), color=MIDS, buff=0.12, stroke_width=5)

        left = VGroup(Text("the underneath:", font_size=19, color=DIM),
                      Text("the leaning,", font_size=19, color=DIM),
                      Text("layer by layer", font_size=19, color=DIM)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(layers, LEFT, buff=0.8)
        surf = VGroup(Text("the surface:", font_size=17, color=DIM, slant=ITALIC),
                      Text("what the agent says", font_size=17, color=DIM, slant=ITALIC)).arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(reply_g, RIGHT, buff=0.45)

        whole = VGroup(glow, layers, reply_g, arr_up, req, probe_g, arr_probe, left, surf)
        whole.scale(0.82).move_to(ORIGIN)
        self.add(whole)


class DotProjection(Scene):
    def construct(self):
        plane = NumberPlane(x_range=[-1,6,1], y_range=[-1,5,1], x_length=8, y_length=6,
                            background_line_style={"stroke_color": ManimColor("#DCE3EA"), "stroke_width":1,
                                                   "stroke_opacity":0.9}).set_opacity(0.9)
        plane.axes.set_stroke(ManimColor("#B8C0C8"), width=1.5)
        o = plane.c2p(0,0)
        d = plane.c2p(4.4,1.7); a = plane.c2p(2.4,3.2)
        import numpy as np
        dv = np.array([4.4,1.7]); av = np.array([2.4,3.2]); pv = (av@dv)/(dv@dv)*dv
        p = plane.c2p(*pv)
        probe = Arrow(o, d, buff=0, color=INK, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        act   = Arrow(o, a, buff=0, color=ACC, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        drop  = DashedLine(a, p, color=GREY, stroke_width=2.5, dash_length=0.12)
        score = Line(o, p, color=GRN, stroke_width=10).set_opacity(0.55)
        dot   = Dot(p, color=GRN, radius=0.08)
        lp = Text("the probe", font_size=26, color=INK, weight=BOLD).next_to(d, RIGHT, buff=0.15)
        la = Text("the request", font_size=26, color=ACC, weight=BOLD).next_to(a, UP, buff=0.15)
        ls = Text("score", font_size=24, color=GRN, weight=BOLD).next_to(score.get_center(), DOWN, buff=0.2)
        cap = Text("the score = how far the request reaches along the probe",
                   font_size=22, color=INK).to_edge(DOWN, buff=0.35)
        g = VGroup(plane, score, probe, act, drop, dot, lp, la, ls).scale(0.9).move_to(ORIGIN+UP*0.3)
        self.add(g, cap)


class Iceberg(Scene):
    def construct(self):
        water = 1.2
        line = DashedLine(LEFT*6, RIGHT*6, color=ManimColor("#5B8DB8"), stroke_width=2).shift(UP*water)
        # tip above water: the words
        tip = Polygon([-1.3,water,0],[1.3,water,0],[0.3,water+1.7,0],[-0.3,water+1.7,0],
                      stroke_color=GRN, fill_color=LGRN, fill_opacity=1, stroke_width=2)
        words = Text('"Your refund\nis processed"', font_size=20, color=INK, line_spacing=0.7).move_to(tip).shift(UP*0.15)
        # mass below: the activation
        mass = Polygon([-1.3,water,0],[1.3,water,0],[3.4,-1.0,0],[2.2,-3.2,0],[-2.2,-3.2,0],[-3.4,-1.0,0],
                       stroke_color=INK, fill_color=LINK, fill_opacity=1, stroke_width=2)
        # faint layer lines in the mass
        lines = VGroup(*[Line([-2.9+0.5*i*0.0-3.0+0.6*k,0,0],[3.0-0.6*k,0,0]) for k,i in enumerate([0])])
        below = VGroup(*[Line(LEFT*(3.1-0.55*k), RIGHT*(3.1-0.55*k), color=ManimColor("#C3CDD8"),
                              stroke_width=1.5).shift(DOWN*(0.2+0.62*k)) for k in range(5)])
        below.set_z_index(1)
        surf = Text("the surface: the words", font_size=22, color=GREY, slant=ITALIC).next_to(tip, RIGHT, buff=0.6).shift(UP*0.4)
        deep = Text("the underneath:\nthe activation we read", font_size=22, color=GREY, slant=ITALIC,
                    line_spacing=0.7).next_to(mass, RIGHT, buff=0.5)
        wl = Text("waterline", font_size=16, color=ManimColor("#5B8DB8")).next_to(line, LEFT, buff=0.15).shift(RIGHT*0.2+UP*0.15)
        g = VGroup(mass, below, tip, words, line, surf, deep, wl).scale(0.9).move_to(ORIGIN)
        self.add(g)


class VirusDefinitions(Scene):
    def construct(self):
        model = RoundedRectangle(width=6.2, height=4.2, corner_radius=0.15, stroke_color=INK,
                                 stroke_width=2.5, fill_color=LINK, fill_opacity=1)
        mlab = VGroup(Text("the model", font_size=30, color=INK, weight=BOLD),
                      Text("~3 GB of weights", font_size=22, color=GREY)).arrange(DOWN, buff=0.15).move_to(model)
        probe = RoundedRectangle(width=1.15, height=0.8, corner_radius=0.08, stroke_color=AMB,
                                 stroke_width=2.5, fill_color=ManimColor("#F7ECD9"), fill_opacity=1)
        plab = VGroup(Text("intent", font_size=15, color=AMB, weight=BOLD),
                      Text("probe", font_size=15, color=AMB, weight=BOLD),
                      Text("16 KB", font_size=12, color=INK)).arrange(DOWN, buff=0.03).move_to(probe)
        probe_g = VGroup(probe, plab)
        grp = VGroup(VGroup(model, mlab), probe_g).arrange(RIGHT, buff=1.4)
        cap = Text("the probe ships beside the model, the way antivirus ships its definitions",
                   font_size=22, color=INK).next_to(grp, DOWN, buff=0.7)
        scale_note = Text("200,000 x smaller", font_size=20, color=AMB, slant=ITALIC).next_to(probe_g, UP, buff=0.4)
        self.add(grp.scale(0.95).move_to(UP*0.3), cap, scale_note)


class DeceptionReveal(Scene):
    def construct(self):
        bubble = RoundedRectangle(width=6.0, height=1.0, corner_radius=0.4, stroke_color=GRN,
                                  stroke_width=2.5, fill_color=LGRN, fill_opacity=1)
        btext = Text('the agent tells the user:  "Done, your request has been processed."',
                     font_size=20, color=INK)
        btext.scale_to_fit_width(bubble.width - 0.4).move_to(bubble)
        top = VGroup(bubble, btext).to_edge(UP, buff=1.0)

        def meter(label, val, color, blind=False):
            bar_bg = RoundedRectangle(width=5.2, height=0.42, corner_radius=0.2,
                                      stroke_color=GREY, stroke_width=1.5, fill_color=LINK, fill_opacity=1)
            fill = RoundedRectangle(width=5.2*val, height=0.42, corner_radius=0.2, stroke_width=0,
                                    fill_color=color, fill_opacity=1).align_to(bar_bg, LEFT)
            num = Text(f"{val:.3f}", font_size=26, color=color, weight=BOLD).next_to(bar_bg, RIGHT, buff=0.3)
            lab = Text(label, font_size=22, color=INK).next_to(bar_bg, UP, buff=0.12).align_to(bar_bg, LEFT)
            tag = Text("blind, like a coin flip" if blind else "still sees the intent",
                       font_size=16, color=GREY, slant=ITALIC).next_to(bar_bg, DOWN, buff=0.1).align_to(bar_bg, LEFT)
            return VGroup(lab, bar_bg, fill, num, tag)

        m1 = meter("reading the reply (what a defender sees)", 0.454, ACC, blind=True)
        m2 = meter("reading the activation (the probe)", 0.993, GRN)
        meters = VGroup(m1, m2).arrange(DOWN, buff=0.9, aligned_edge=LEFT).next_to(top, DOWN, buff=0.9)
        self.add(top, meters)
