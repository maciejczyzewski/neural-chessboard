import os, glob

files = glob.glob('test/in/*.jpg')
for fname in files:
	fname = fname.replace('.jpg', '').replace('test/in/', '')
	os.system('./fen ' + fname)

# import chess, chess.svg

ARR_FEN = [
'r1b5/pp3p1p/4kQp1/2pNN3/2PnP3/6P1/PP4qP/3RK2R w - -',
'1r4k1/2pb1ppp/4p2n/p1qp4/1n6/2PBPNP1/1PKB1P1P/3R3R w - -',
'r3r1k1/pp3ppp/8/3P4/2PBn1b1/2P5/P3BPPP/R3K2R w - -',
'q3k2B/3p1p1p/rp3N2/p5p1/1p4P1/1P1P3P/P4K2/R2Q3R w - -',
'rn1R1k1r/pp3ppp/2p5/4Q1B1/4n3/P1P5/2q2PPP/5RK1 w - -',
'rnbqkbnr/pppp2pp/5p2/4p3/4P3/5P2/PPPP2PP/RNBQKBNR w - -',
'q3k2B/3p1p1p/rp3N2/p5p1/1p4P1/1P1P3P/P4K2/R2Q3R w - -',
'r3q2r/pp2bppp/2p5/4n2K/5Nb1/6N1/PP4PP/2R2B1R w - -',
'r2qr2k/ppp1n1pp/7n/3p1bN1/1P1P4/1BNR4/1PQ2PPP/4R1K1 w - -',
'4B3/3P1R2/4k3/1P1Q4/1N4P1/8/8/6K1 w - -',
'r3k2r/pb1n2b1/1ppp1q1p/4pp2/1N6/3PQNP1/PPP2PBP/3RR1K1 w - -',
'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',
'4n1rk/p3np1p/q1r2p2/1pp1p2N/2PpP1P1/1P1P4/P2Q1RB1/5RK1 w - -',
'8/4Q3/3p4/6R1/6P1/5k2/1KPrp3/8 w - -',
'5r2/ppp4P/3kb1R1/3pb3/8/2PB4/P6P/4R1K1 w - -',
'1r3k1r/1b1N1npp/pp6/8/P3N3/4P1P1/1BPP1P1P/RR3K2 w - -',
'2r2rk1/pp1n1pbp/3p2p1/2q5/2PN4/1P1Q2PP/PB3P2/R4RK1 w - -',
'r2qk2r/ppp2ppp/2np4/2b1p2n/2B1P3/P6P/1PPPQPP1/RNB2RK1 w - -',
'r2qkb1r/pp3ppp/2n1pn2/3p1b2/3P4/P1N2NP1/1P2PPBP/R1BQK2R w - -',
'rnb1kb1r/pp3pp1/4pn1p/2P5/4p2B/8/qPPNQPPP/2KR1BNR w - -'
]

"""
for key, fen in enumerate(ARR_FEN):
	board = chess.Board(fen + " 1 1")
	print("ID", key + 1); print(board)
	svg = chess.svg.board(board=board)
	ffile = open("test/out/fen_{}.svg".format(key + 1), 'w')
	ffile.write(svg)
	ffile.close()
	print("---")
"""
