import sqlalchemy as db

usr_factors = 'x8313'
pswd_factors = 'animalcheese'
db_factors = 'x8_factors'

engine_factors = db.create_engine(
    'postgresql://{0}:{1}@x8-factors.c4umbmfqfrd8.us-east-2.rds.amazonaws.com:5433/{2}'
    .format(usr_factors, pswd_factors, db_factors)
)

# engine ac credentials
usr_ac = 'ac_2012'
pswd_ac = 'suspicious'
db_ac = 'ac'

# DEPRECATED
# original ac connection (Dave's)
# engine_ac = db.create_engine(
#     'postgresql://{0}:{1}@dev.tennis-edge.com:5432/{2}'
#     .format(usr_ac, pswd_ac, db_ac)
# )

# ac AWS RDS credentials (clone of engine_ac)
engine_ac = db.create_engine(
    'postgresql://{0}:{1}@animal-crackers.c4umbmfqfrd8.us-east-2.rds.amazonaws.com:5432/{2}'
    .format(usr_ac, pswd_ac, db_ac)
)


def makesqllist(id_list):
    '''turn list into string with single quotes around elements and commas between elements'''
    idList=",".join(["'"+x+"'" for x in id_list])
    return idList
