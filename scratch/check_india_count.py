import duckdb, time

con = duckdb.connect()
s1 = '6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv'
s2 = '6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source2.tsv'
s3 = '6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source3.tsv'

in_s1 = con.execute(f"SELECT count(*) FROM read_csv('{s1}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='India'").fetchone()[0]
in_s2 = con.execute(f"SELECT count(*) FROM read_csv('{s2}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='India'").fetchone()[0]
in_s3 = con.execute(f"SELECT count(*) FROM read_csv('{s3}', delim='\\t', header=True, quote='', all_varchar=True) WHERE country='India'").fetchone()[0]

print(f"India S1 count: {in_s1:,}")
print(f"India S2 count: {in_s2:,}")
print(f"India S3 count: {in_s3:,}")
print(f"Total India Targets: {in_s2 + in_s3:,}")
