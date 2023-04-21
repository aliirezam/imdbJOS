!pip install -U Pillow==6.2.2

!pip install torch==1.0.0 torchvision==0.2.1

!pip install torchfold

!pip install -U psqlparse==1.0rc7

!pip install tf-estimator-nightly

try:
  %tensorflow_version 2.x
except Exception:
  pass

!pip install tensorflow-io

!sudo apt-get -y -qq update
!sudo apt-get -y -qq install postgresql
!sudo service postgresql start

# ===>  CREATE DATABASE USER & PASSWORD:

!sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD '12345678';"

# ===> CREATE DATABASE NAME

!sudo -u postgres psql -U postgres -c 'DROP DATABASE IF EXISTS imdbload;'
!sudo -u postgres psql -U postgres -c 'CREATE DATABASE imdbload;'

# ===> CONNECTED TO DATABASE FOR CREATE TABLE:

%env dbName=imdbload
%env ip=localhost
%env port=5432
%env userName=postgres
%env password=12345678

!pip install psycopg2-binary

!pip install wget

!wget http://homepages.cwi.nl/~boncz/job/imdb.tgz

!sudo -u postgres psql -U postgres -d imdbload -c "COPY aka_name FROM '$(pwd)/imdb-datasets-ftp/aka_name.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY aka_title FROM '$(pwd)/imdb-datasets-ftp/aka_title.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY cast_info FROM '$(pwd)/imdb-datasets-ftp/cast_info.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY char_name FROM '$(pwd)/imdb-datasets-ftp/char_name.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY comp_cast_type FROM '$(pwd)/imdb-datasets-ftp/comp_cast_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY company_name FROM '$(pwd)/imdb-datasets-ftp/company_name.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY company_type FROM '$(pwd)/imdb-datasets-ftp/company_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY complete_cast FROM '$(pwd)/imdb-datasets-ftp/complete_cast.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY info_type FROM '$(pwd)/imdb-datasets-ftp/info_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY keyword FROM '$(pwd)/imdb-datasets-ftp/keyword.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY kind_type FROM '$(pwd)/imdb-datasets-ftp/kind_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY link_type FROM '$(pwd)/imdb-datasets-ftp/link_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY movie_companies FROM '$(pwd)/imdb-datasets-ftp/movie_companies.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY movie_info FROM '$(pwd)/imdb-datasets-ftp/movie_info.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY movie_info_idx FROM '$(pwd)/imdb-datasets-ftp/movie_info_idx.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY movie_keyword FROM '$(pwd)/imdb-datasets-ftp/movie_keyword.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY movie_link FROM '$(pwd)/imdb-datasets-ftp/movie_link.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY name FROM '$(pwd)/imdb-datasets-ftp/name.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY person_info FROM '$(pwd)/imdb-datasets-ftp/person_info.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY role_type FROM '$(pwd)/imdb-datasets-ftp/role_type.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
!sudo -u postgres psql -U postgres -d imdbload -c "COPY title FROM '$(pwd)/imdb-datasets-ftp/title.csv' WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";

try:
    con = psycopg2.connect(database="imdbload",
        user="postgres",
        password="12345678",
        host="localhost",
        port="5432")
    print("Connection to database succeeded...")
except:
    print("Connection to database failed...")
	
cur = con.cursor()

class Config:
    def __init__(self,):
        self.sytheticDir = "Queries/sytheic"
        self.JOBDir = "JOB-queries"
        self.schemaFile = "schema.sql"
        self.dbName = "imdbload"
        self.userName = "postgres"
        self.password = "12345678"
        self.ip = "localhost"
        self.port = 5432
		
import numpy as np
class Expr:
    def __init__(self, expr,list_kind = 0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0
    def isCol(self,):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def getValue(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"]+"\'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise "unknown Value in Expr"
        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]['typeName']['TypeName']['names'])==1:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][0]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+"'"
            else:
                if value_expr["TypeCast"]['typeName']['TypeName']['typmods'][0]['A_Const']['val']['Integer']['ival']==2:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' month"
                else:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' year"
        else:
            print(value_expr.keys())
            raise "unknown Value in Expr"

    def getAliasName(self,):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(self,):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(self,):
        if self.isCol():
            return self.getAliasName()+"."+self.getColumnName()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "("+",\n".join([self.getValue(x) for x in self.expr])+")"
            elif self.list_kind == 10:
                return " AND ".join([self.getValue(x) for x in self.expr])
            else:
                raise "list kind error"

        else:
            raise "No Known type of Expr"


class TargetTable:
    def __init__(self, target):
        """
        {'location': 7, 'name': 'alternative_name', 'val': {'FuncCall': {'funcname': [{'String': {'str': 'min'}}], 'args': [{'ColumnRef': {'fields': [{'String': {'str': 'an'}}, {'String': {'str': 'name'}}], 'location': 11}}], 'location': 7}}}
        """
        self.target = target
    #         print(self.target)

    def getValue(self,):
        columnRef = self.target["val"]["FuncCall"]["args"][0]["ColumnRef"]["fields"]
        return columnRef[0]["String"]["str"]+"."+columnRef[1]["String"]["str"]

    def __str__(self,):
        try:
            return self.target["val"]["FuncCall"]["funcname"][0]["String"]["str"]+"(" + self.getValue() + ")" + " AS " + self.target['name']
        except:
            if "FuncCall" in self.target["val"]:
                return "count(*)"
            else:
                return "*"

class FromTable:
    def __init__(self, from_table):
        """
        {'alias': {'Alias': {'aliasname': 'an'}}, 'location': 168, 'inhOpt': 2, 'relpersistence': 'p', 'relname': 'aka_name'}
        """
        self.from_table = from_table

    def getFullName(self,):
        return self.from_table["relname"]

    def getAliasName(self,):
        return self.from_table["alias"]["Alias"]["aliasname"]

    def __str__(self,):
        return self.getFullName()+" AS "+self.getAliasName()


class Comparison:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"],self.kind)
            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x)
                              for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.aliasname_list.append(comp.lexpr.getAliasName())
                    self.lexpr = comp.lexpr
                    self.column_list.append(comp.lexpr.getColumnName())
                    break
            self.comp_kind = 2
    def isCol(self,):
        return False
    def __str__(self,):

        if self.comp_kind == 0:
            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"]=="!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise "Operation ERROR"
            return str(self.lexpr)+" "+Op+" "+str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr)+" IS NOT NULL"
            else:
                return str(self.lexpr)+" IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( "+str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res

class Table:
    def __init__(self, table_tree):
        self.name = table_tree["relation"]["RangeVar"]["relname"]
        self.column2idx = {}
        self.idx2column = {}
        for idx, columndef in enumerate(table_tree["tableElts"]):
            self.column2idx[columndef["ColumnDef"]["colname"]] = idx
            self.idx2column[idx] = columndef["ColumnDef"]["colname"]

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))


class DB:
    def __init__(self, schema,TREE_NUM_IN_NET=40):
        from psqlparse import parse_dict
        parse_tree = parse_dict(schema)

        self.tables = []
        self.name2idx = {}
        self.table_names = []
        self.name2table = {}
        self.size = 0
        self.TREE_NUM_IN_NET = TREE_NUM_IN_NET
        for idx, table_tree in enumerate(parse_tree):
            self.tables.append(Table(table_tree["CreateStmt"]))
            self.table_names.append(self.tables[-1].name)
            self.name2idx[self.tables[-1].name] = idx
            self.name2table[self.tables[-1].name] = self.tables[-1]

        self.columns_total = 0

        for table in self.tables:
            self.columns_total += len(table.idx2column)

        self.size = len(self.table_names)

    def __len__(self,):
        if self.size == 0:
            self.size = len(self.table_names)
        return self.size

    def oneHotAll(self,):
        return np.zeros((1, self.size))

    def network_size(self,):
        return self.TREE_NUM_IN_NET*self.size


import psycopg2
import json
from math import log
class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None

class PGRunner:
    def __init__(self,dbname = '',user = '',password = '',host = '',port = '',isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        """
        :param dbname:
        :param user:
        :param password:
        :param host:
        :param port:
        :param latencyRecord:-1:loadFromFile
        :param latencyRecordFile:
        """
        self.con = psycopg2.connect(database=dbname, user=user,
                               password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.config = PGConfig()
        self.isLatencyRecord = latencyRecord
        # self.LatencyRecordFileHandle = None
        global LatencyRecordFileHandle
        self.isCostTraining = isCostTraining
        if latencyRecord:
            LatencyRecordFileHandle = self.generateLatencyPool(latencyRecordFile)


    def generateLatencyPool(self,fileName):
        """
        :param fileName:
        :return:
        """
        import os
        import json
        if os.path.exists(fileName):
            f = open(fileName,"r")
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                global LatencyDict
                LatencyDict[data[0]] = data[1]
            f = open(fileName,"a")
        else:
            f = open(fileName,"w")
        return f
    def getLatency(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        # query = sql.toSql()
        if self.isCostTraining:
            return self.getCost(sql,sqlwithplan)
        global LatencyDict
        if self.isLatencyRecord:
            if sqlwithplan in LatencyDict:
                return LatencyDict[sqlwithplan]

        self.cur.execute("set join_collapse_limit = 1;")
        self.cur.execute("SET statement_timeout = "+str(int(sql.timeout()))+ ";")
        self.cur.execute("set max_parallel_workers=1;")
        self.cur.execute("set max_parallel_workers_per_gather = 1;")
        self.cur.execute("set geqo_threshold = 20;")
        self.cur.execute("EXPLAIN "+sqlwithplan)
        thisQueryCost = self.getCost(sql,sqlwithplan)
        if thisQueryCost / sql.getDPCost()<100:
            try:
                self.cur.execute("EXPLAIN ANALYZE "+sqlwithplan)
                rows = self.cur.fetchall()
                row = rows[0][0]
                afterCost = float(rows[0][0].split("actual time=")[1].split("..")[1].split(" ")[
                                      0])
            except:
                self.con.commit()
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost
    def getCost(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        self.cur.execute("set join_collapse_limit = 1;")
        self.cur.execute("set max_parallel_workers=1;")
        self.cur.execute("set max_parallel_workers_per_gather = 1;")
        self.cur.execute("set geqo_threshold = 20;")
        self.cur.execute("SET statement_timeout =  100000;")

        self.cur.execute("EXPLAIN "+sqlwithplan)
        rows = self.cur.fetchall()
        row = rows[0][0]
        afterCost = float(rows[0][0].split("cost=")[1].split("..")[1].split(" ")[
                              0])
        self.con.commit()
        return afterCost

    def getDPPlanTime(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the planTime of sql
        """
        import time
        startTime = time.time()
        cost = self.getCost(sql,sqlwithplan)
        plTime = time.time()-startTime
        return plTime
    def getSelectivity(self,table,whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]
        self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        totalQuery = "select * from "+table+";"
        #     print(totalQuery)

        self.cur.execute("EXPLAIN "+totalQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        # print(resQuery)
        self.cur.execute("EXPLAIN  "+resQuery)
        rows = self.cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        #     print(stored_selectivity_fake[whereCondition],select_rows,total_rows)
        return selectivityDict[whereCondition]
		

# from JOBParser import TargetTable,FromTable,Comparison
max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np

tree_lstm_memory = {}
class JoinTree:
    def __init__(self,sqlt,db_info,pgRunner,device):
        from psqlparse import parse_dict
        global tree_lstm_memory
        tree_lstm_memory  ={}
        self.sqlt = sqlt
        self.sql = self.sqlt.sql
        parse_result = parse_dict(self.sql)[0]["SelectStmt"]
        self.target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        self.from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        self.aliasname2fullname = {}
        self.pgrunner = pgRunner
        self.device = device
        self.aliasname2fromtable={}
        for table in self.from_table_list:
            self.aliasname2fromtable[table.getAliasName()] = table
            self.aliasname2fullname[table.getAliasName()] = table.getFullName()
        self.aliasnames = set(self.aliasname2fromtable.keys())
        self.comparison_list =[Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        self.db_info = db_info
        self.join_list = {}
        self.filter_list = {}

        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])
        self.aliasnames_fa = {}
        self.aliasnames_set = {}
        self.aliasnames_join_set = {}
        self.left_son = {}
        self.right_son = {}
        self.total = 0
        self.left_aliasname = {}
        self.right_aliasname = {}

        self.table_fea_set = {}
        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = [0.0]*max_column_in_table*2

        ##提取所有的Join和filter
        self.join_candidate = set()
        self.join_matrix=[]
        for aliasname in self.aliasnames_root_set:
            self.join_list[aliasname] = []
        for idx in range(len(self.db_info)):
            self.join_matrix.append([0]*len(self.db_info))
        for comparison in self.comparison_list:
            if len(comparison.aliasname_list) == 2:
                if not comparison.aliasname_list[0] in self.join_list:
                    self.join_list[comparison.aliasname_list[0]] = []
                if not comparison.aliasname_list[1] in self.join_list:
                    self.join_list[comparison.aliasname_list[1]] = []
                self.join_list[comparison.aliasname_list[0]].append((comparison.aliasname_list[1],comparison))
                left_aliasname = comparison.aliasname_list[0]
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]
                table_idx = left_table_class.column2idx[comparison.column_list[0]]
                self.table_fea_set[left_aliasname][table_idx * 2] = 1
                self.join_list[comparison.aliasname_list[1]].append((comparison.aliasname_list[0],comparison))
                right_aliasname = comparison.aliasname_list[1]
                right_fullname = self.aliasname2fullname[right_aliasname]
                right_table_class = db_info.name2table[right_fullname]
                table_idx = right_table_class.column2idx[comparison.column_list[1]]
                self.table_fea_set[right_aliasname][table_idx * 2] = 1
                self.join_candidate.add((comparison.aliasname_list[0],comparison.aliasname_list[1]))
                self.join_candidate.add((comparison.aliasname_list[1],comparison.aliasname_list[0]))
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            else:
                if not comparison.aliasname_list[0] in self.filter_list:
                    self.filter_list[comparison.aliasname_list[0]] = []
                self.filter_list[comparison.aliasname_list[0]].append(comparison)
                left_aliasname = comparison.aliasname_list[0]
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]
                table_idx = left_table_class.column2idx[comparison.column_list[0]]
                self.table_fea_set[left_aliasname][table_idx * 2 + 1] += self.pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))


        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = self.device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])


        predice_list_dict={}
        for table in self.db_info.tables:
            predice_list_dict[table.name] = [0] * len(table.column2idx)
        for filter_table in self.filter_list:
            for comparison in self.filter_list[filter_table]:
                aliasname = comparison.aliasname_list[0]
                fullname = self.aliasname2fullname[aliasname]
                table = self.db_info.name2table[fullname]
                for column in comparison.column_list:
                    columnidx = table.column2idx[column]
                    predice_list_dict[self.aliasname2fullname[filter_table]][columnidx] = 1
        self.predice_feature = []
        for fullname in predice_list_dict:
            self.predice_feature+= predice_list_dict[fullname]
        self.predice_feature = np.asarray(self.predice_feature).reshape(1,-1)
        self.join_matrix = torch.tensor(np.asarray(self.join_matrix).reshape(1,-1),device = self.device,dtype = torch.float32)

    def resetJoin(self):
        self.aliasnames_fa = {}
        self.left_son = {}
        self.right_son = {}
        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])

        self.left_aliasname  = {}
        self.right_aliasname =  {}
        self.aliasnames_join_set = {}
        for aliasname in self.aliasnames_root_set:
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])

        self.total = 0
    def findFather(self,node_name):
        fa_name = node_name
        while  fa_name in self.aliasnames_fa:
            fa_name = self.aliasnames_fa[fa_name]
        while  node_name in self.aliasnames_fa:
            temp_name = self.aliasnames_fa[node_name]
            self.aliasnames_fa[node_name] = fa_name
            node_name = temp_name
        return fa_name

    def joinTables(self,aliasname_left,aliasname_right,fake=False):
        aliasname_left_fa = self.findFather(aliasname_left)
        aliasname_right_fa = self.findFather(aliasname_right)
        self.aliasnames_fa[aliasname_left_fa] = self.total
        self.aliasnames_fa[aliasname_right_fa] = self.total
        self.left_son[self.total] = aliasname_left_fa
        self.right_son[self.total] = aliasname_right_fa
        self.aliasnames_root_set.add(self.total)

        self.left_aliasname[self.total] = aliasname_left
        self.right_aliasname[self.total] = aliasname_right
        if not fake:
            self.aliasnames_set[self.total] = self.aliasnames_set[aliasname_left_fa]|self.aliasnames_set[aliasname_right_fa]
            self.aliasnames_join_set[self.total] = (self.aliasnames_join_set[aliasname_left_fa]|self.aliasnames_join_set[aliasname_right_fa])-self.aliasnames_set[self.total]
            self.aliasnames_root_set.remove(aliasname_left_fa)
            self.aliasnames_root_set.remove(aliasname_right_fa)

        self.total += 1
    def recTable(self,node):
        if isinstance(node,int):
            res =  "("
            leftRes = self.recTable(self.left_son[node])
            if not self.left_son[node] in self.aliasnames:
                leftRes = leftRes[1:-1]

            res += leftRes + "\n"
            filter_list = []
            on_list = []
            if self.left_son[node] in self.filter_list:
                for condition in self.filter_list[self.left_son[node]]:
                    filter_list.append(str(condition))

            if self.right_son[node] in self.filter_list :
                for condition in self.filter_list[self.right_son[node]]:
                    filter_list.append(str(condition))

            cpList = []
            joined_aliasname = set([self.left_aliasname[node],self.right_aliasname[node]])
            for left_table in self.aliasnames_set[self.left_son[node]]:
                for right_table,comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[self.right_son[node]]:
                        if (comparison.aliasname_list[1] in joined_aliasname and comparison.aliasname_list[0] in joined_aliasname):
                            cpList.append(str(comparison))
                        else:
                            on_list.append(str(comparison))
            if len(filter_list+on_list+cpList)>0:
                res += "inner join "
                res += self.recTable(self.right_son[node])
                res += "\non "
                res += " AND ".join(cpList + on_list+filter_list)
            else:
                res += "cross join "
                res += self.recTable(self.right_son[node])

            res += ")"
            return res
        else:
            return str(self.aliasname2fromtable[node])
    def encode_tree_regular(self,model, node_idx):

        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            left_emb =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25],device = self.device),self.table_fea_set[left_aliasname])
            right_emb = model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25],device = self.device),self.table_fea_set[right_aliasname])
            return model.inputX(left_emb[0],right_emb[0])
        def encode_node(node):
            if node in tree_lstm_memory:
                return tree_lstm_memory[node]
            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                res =  model.childrenNode(left_h, left_c, right_h, right_c,inputX)
                if self.total > node + 1:
                    tree_lstm_memory[node] = res
            else:
                res =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[node]]],device = self.device),self.table_fea_set[node])
                tree_lstm_memory[node] = res

            return res
        encoding, _ = encode_node(node_idx)
        return encoding
    def encode_tree_fold(self,fold, node_idx):
        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            left_emb,c1 =  fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25,self.table_fea_set[left_aliasname]).split(2)
            right_emb,c2 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25,self.table_fea_set[right_aliasname]).split(2)
            return fold.add('inputX',left_emb,right_emb)
        def encode_node(node):

            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                return fold.add('childrenNode',left_h, left_c, right_h, right_c,inputX).split(2)
            else:
                return fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[node]],self.table_fea_set[node]).split(2)
            return None
        encoding, _ = encode_node(node_idx)
        return encoding
    def toSql(self,):
        root = self.total - 1
        res = "select "+",\n".join([str(x) for x in self.target_table_list])+"\n"
        res  += "from " + self.recTable(root)[1:-1]
        res += ";"
        return res
    def plan2Cost(self):
        sql = self.toSql()
        return self.pgrunner.getLatency(self.sqlt,sql)

class sqlInfo:
    def __init__(self,pgRunner,sql,filename):
        self.DPLantency = None
        self.DPCost = None
        self.bestLatency = None
        self.bestCost = None
        self.bestOrder = None
        self.plTime = None
        self.pgRunner = pgRunner
        self.sql = sql
        self.filename = filename
    def getDPlantecy(self,):
        if self.DPLantency == None:
            self.DPLantency = self.pgRunner.getLatency(self,self.sql)
        return self.DPLantency
    def getDPPlantime(self,):
        if self.plTime == None:
            self.plTime = self.pgRunner.getDPPlanTime(self,self.sql)
        return self.plTime
    def getDPCost(self,):
        if self.DPCost == None:
            self.DPCost = self.pgRunner.getCost(self,self.sql)
        return self.DPCost
    def timeout(self,):
        if self.DPLantency == None:
            return 1000000
        return self.getDPlantecy()*4+self.getDPPlantime()
    def getBestOrder(self,):
        return self.bestOrder
    def updateBestOrder(self,latency,order):
        if self.bestOrder == None or self.bestLatency > latency:
            self.bestLatency = latency
            self.bestOrder = order


import torch
from torch.nn import init
import torchfold
import torch.nn as nn
class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.FC1 = nn.Linear(num_units, 5 * num_units)
        self.FC2 = nn.Linear(num_units, 5 * num_units)
        self.FC0 = nn.Linear(num_units, 5 * num_units)
        self.LNh = nn.LayerNorm(num_units,)
        self.LNc = nn.LayerNorm(num_units,)
    def forward(self, left_in, right_in,inputX):
        lstm_in = self.FC1(left_in[0])
        lstm_in += self.FC2(right_in[0])
        lstm_in += self.FC0(inputX)
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        return h,c
class TreeRoot(nn.Module):
    def __init__(self,num_units):
        super(TreeRoot, self).__init__()
        self.num_units = num_units
        self.FC = nn.Linear(num_units, num_units)
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        # self.max_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, tree_list):

        return self.relu(self.FC(self.sum_pooling(tree_list)).view(-1,self.num_units))

class SPINN(nn.Module):

    def __init__(self, n_classes, size, n_words, mask_size,device,max_column_in_table = 15):
        super(SPINN, self).__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.tree_root = TreeRoot(size)
        self.FC = nn.Linear(size*2, size)
        self.table_embeddings = nn.Embedding(n_words, size)#2 * max_column_in_table * size)
        self.column_embeddings = nn.Embedding(n_words, 2 * max_column_in_table * size)
        self.out = nn.Linear(size*2, size)
        self.out2 = nn.Linear(size, n_classes)
        self.outFc = nn.Linear(mask_size, size)
        self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.leafFC = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()
        self.LN1 = nn.LayerNorm(size,)
        self.LN2 = nn.LayerNorm(size,)
        self.max_column_in_table = max_column_in_table
        self.leafLn = nn.LayerNorm(size,)
        self.device = device

    def leaf(self, word_id, table_fea=None):
        all_columns = table_fea.view(-1,self.max_column_in_table*2,1)*self.column_embeddings(word_id).reshape(-1,2 * self.max_column_in_table,self.size)
        all_columns = self.relu(self.leafFC(all_columns))
        table_emb = self.max_pooling(all_columns.view(-1,self.max_column_in_table*2,self.size)).view(-1,self.size)
        return self.leafLn(table_emb), torch.zeros(word_id.size()[0], self.size,device = self.device,dtype = torch.float32)
    def inputX(self,left_emb,right_emb):
        cat_emb = torch.cat([left_emb,right_emb],dim = 1)
        return self.relu(self.FC(cat_emb))
    def childrenNode(self, left_h, left_c, right_h, right_c,inputX):
        return self.tree_lstm((left_h, left_c), (right_h, right_c),inputX)
    def root(self,tree_list):
        return self.tree_root(tree_list).view(-1,self.size)
    def logits(self, encoding,join_matrix):
        encoding = self.root(encoding.view(1,-1,self.size))
        matrix = self.relu(self.outFc(join_matrix))
        outencoding = torch.cat([encoding,matrix],dim = 1)
        return self.out2(self.relu(self.out(outencoding)))
		
		
import math
import random
import torchfold
import torch.nn.functional as F
import torchvision.transforms as T
import torch
from collections import namedtuple
# from sqlSample import JoinTree
import torch.optim as optim
import numpy as np
from math import log
from itertools import count

class ENV(object):
    def __init__(self,sql,db_info,pgrunner,device):
        self.sel = JoinTree(sql,db_info,pgrunner,device )
        self.sql = sql
        self.hashs = ""
        self.table_set = set([])
        self.res_table = []
        self.init_table = None
        self.planSpace = 0#0:leftDeep,1:bushy

    def actionValue(self,left,right,model):
        self.sel.joinTables(left,right,fake = True)
        res_Value = self.selectValue(model)
        self.sel.total -= 1
        self.sel.aliasnames_root_set.remove(self.sel.total)
        self.sel.aliasnames_fa.pop(self.sel.left_son[self.sel.total])
        self.sel.aliasnames_fa.pop(self.sel.right_son[self.sel.total])
        return res_Value

    def selectValue(self,model):
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
                tree_state.append(self.sel.encode_tree_regular(model,idx))
        res = torch.cat(tree_state,dim = 0)
        return model.logits(res,self.sel.join_matrix)

    def selectValueFold(self,fold):
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
                tree_state.append(self.sel.encode_tree_fold(fold,idx))
            #         res = torch.cat(tree_state,dim = 0)
        return tree_state
        return fold.add('logits',tree_state,self.sel.join_matrix)



    def takeAction(self,left,right):
        self.sel.joinTables(left,right)
        self.hashs += left
        self.hashs += right
        self.hashs += " "

    def hashcode(self):
        return self.sql.sql+self.hashs
    def allAction(self,model):
        action_value_list = []
        for one_join in self.sel.join_candidate:
            l_fa = self.sel.findFather(one_join[0])
            r_fa  =self.sel.findFather(one_join[1])
            if self.planSpace ==0:
                flag1 = one_join[1] ==r_fa and l_fa !=one_join[0]
                if l_fa!=r_fa and (self.sel.total == 0 or flag1):
                    action_value_list.append((self.actionValue(one_join[0],one_join[1],model),one_join))
            elif self.planSpace==1:
                if l_fa!=r_fa:
                    action_value_list.append((self.actionValue(one_join[0],one_join[1],model),one_join))
        return action_value_list
    def reward(self,):
        if self.sel.total+1 == len(self.sel.from_table_list):
            return log( self.sel.plan2Cost())/log(1.5), True
        else:
            return 0,False



Transition = namedtuple('Transition',
                        ('env', 'next_value', 'this_value'))
# bestJoinTreeValue = {}
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.bestJoinTreeValue = {}
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  Transition(*args)
        hashv = data.env.hashcode()
        next_value = data.next_value
        if hashv in self.bestJoinTreeValue and self.bestJoinTreeValue[hashv]<data.this_value:
            if self.bestJoinTreeValue[hashv]<next_value:
                next_value = self.bestJoinTreeValue[hashv]
        else:
            self.bestJoinTreeValue[hashv]  = data.this_value
        data = Transition(data.env,self.bestJoinTreeValue[hashv],data.this_value)
        position = self.position
        self.memory[position] = data
        #         self.position
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory

    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
    def resetbest(self):
        self.bestJoinTreeValue = {}

class DQN:
    def __init__(self,policy_net,target_net,db_info,pgrunner,device):
        self.Memory = ReplayMemory(1000)
        self.BATCH_SIZE = 1

        self.optimizer = optim.Adam(policy_net.parameters(),lr = 3e-4   ,betas=(0.9,0.999))

        self.steps_done = 0
        self.max_action = 25
        self.EPS_START = 0.4
        self.EPS_END = 0.2
        self.EPS_DECAY = 400
        self.policy_net = policy_net
        self.target_net = target_net
        self.db_info = db_info
        self.pgrunner = pgrunner
        self.device = device
        self.steps_done = 0
    def select_action(self, env, need_random = True):

        sample = random.random()
        if need_random:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                      math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = -1
        action_list = env.allAction(self.policy_net)
        action_batch = torch.cat([x[0] for x in action_list],dim = 1)

        if sample > eps_threshold:
            return action_batch,action_list[torch.argmin(action_batch,dim = 1)[0]][1],[x[1] for x in action_list]
        else:
            return action_batch,action_list[random.randint(0,len(action_list)-1)][1],[x[1] for x in action_list]


    def validate(self,val_list, tryTimes = 1):
        rewards = []
        prt = []
        mes = 0
        for sql in val_list:
            pg_cost = sql.getDPlantecy()
            env = ENV(sql,self.db_info,self.pgrunner,self.device)

            for t in count():
                action_list, chosen_action,all_action = self.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                reward, done = env.reward()
                if done:
                    rewards.append(np.exp(reward*log(1.5)-log(pg_cost)))
                    mes = mes + reward*log(1.5)-log(pg_cost)
                    break
        lr = len(rewards)
        from math import e
        print("MRC",sum(rewards)/lr,"GMRL",e**(mes/lr))
        return sum(rewards)/lr

    def optimize_model(self,):
        import time
        startTime = time.time()
        samples = self.Memory.sample(64)
        value_now_list = []
        next_value_list = []
        if (len(samples)==0):
            return
        fold = torchfold.Fold(cuda=True)
        nowL = []
        for one_sample in samples:
            nowList = one_sample.env.selectValueFold(fold)
            nowL.append(len(nowList))
            value_now_list+=nowList
        res = fold.apply(self.policy_net, [value_now_list])[0]
        total = 0
        value_now_list = []
        next_value_list = []
        for idx,one_sample in enumerate(samples):
            value_now_list.append(self.policy_net.logits(res[total:total+nowL[idx]] , one_sample.env.sel.join_matrix ))
            next_value_list.append(one_sample.next_value)
            total += nowL[idx]
        value_now = torch.cat(value_now_list,dim = 0)
        next_value = torch.cat(next_value_list,dim = 0)
        endTime = time.time()
        if True:
            loss = F.smooth_l1_loss(value_now,next_value,size_average=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return None


# from PGUtils import PGRunner
# from sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
# from DQN import DQN,ENV
# from TreeLSTM import SPINN
# from JOBParser import DB
import copy
import torch
from torch.nn import init
# from ImportantConfig import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)

featureSize = 128

policy_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
target_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
for name, param in policy_net.named_parameters():
    print(name,param.shape)
    if len(param.shape)==2:
        init.xavier_normal(param)
    else:
        init.uniform(param)

# policy_net.load_state_dict(torch.load("JOB_tc.pth"))#load cost train model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port,isCostTraining=True,latencyRecord = False,latencyRecordFile = "Cost.json")

DQN = DQN(policy_net,target_net,db_info,pgrunner,device)

def k_fold(input_list,k,ix = 0):
    li = len(input_list)
    kl = (li-1)//k + 1
    train = []
    validate = []
    for idx in range(li):

        if idx%k == ix:
            validate.append(input_list[idx])
        else:
            train.append(input_list[idx])
    return train,validate


def QueryLoader(QueryDir):
    def file_name(file_dir):
        import os
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.sql':
                    L.append(os.path.join(root, file))
        return L
    files = file_name(QueryDir)
    sql_list = []
    for filename in files:
        with open(filename, "r") as f:
            data = f.readlines()
            one_sql = "".join(data)
            sql_list.append(sqlInfo(pgrunner,one_sql,filename))
    return sql_list

def resample_sql(sql_list):
    rewards = []
    reward_sum = 0
    rewardsP = []
    mes = 0
    for sql in sql_list:
        #         sql = val_list[i_episode%len(train_list)]
        pg_cost = sql.getDPlantecy()
        #         continue
        env = ENV(sql,db_info,pgrunner,device)

        for t in count():
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=False)

            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)

            reward, done = env.reward()
            if done:
                mrc = max(np.exp(reward*log(1.5))/pg_cost-1,0)
                rewardsP.append(np.exp(reward*log(1.5)-log(pg_cost)))
                mes += reward*log(1.5)-log(pg_cost)
                rewards.append((mrc,sql))
                reward_sum += mrc
                break
    import random
    print(rewardsP)
    res_sql = []
    print(mes/len(sql_list))
    for idx in range(len(sql_list)):
        rd = random.random()*reward_sum
        for ts in range(len(sql_list)):
            rd -= rewards[ts][0]
            if rd<0:
                res_sql.append(rewards[ts][1])
                break
    return res_sql+sql_list
def train(trainSet,validateSet):

    trainSet_temp = trainSet
    losses = []
    startTime = time.time()
    print_every = 20
    TARGET_UPDATE = 3
    for i_episode in range(0,10000):
        if i_episode % 200 == 100:
            trainSet = resample_sql(trainSet_temp)
        #     sql = random.sample(train_list_back,1)[0][0]
        sqlt = random.sample(trainSet[0:],1)[0]
        pg_cost = sqlt.getDPlantecy()
        env = ENV(sqlt,db_info,pgrunner,device)

        previous_state_list = []
        action_this_epi = []
        nr = True
        nr = random.random()>0.3 or sqlt.getBestOrder()==None
        acBest = (not nr) and random.random()>0.7
        for t in count():
            # beginTime = time.time();
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=nr)
            value_now = env.selectValue(policy_net)
            next_value = torch.min(action_list).detach()
            # e1Time = time.time()
            env_now = copy.deepcopy(env)
            # endTime = time.time()
            # print("make",endTime-startTime,endTime-e1Time)
            if acBest:
                chosen_action = sqlt.getBestOrder()[t]
            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)
            action_this_epi.append((left,right))

            reward, done = env.reward()
            reward = torch.tensor([reward], device=device, dtype = torch.float32).view(-1,1)

            previous_state_list.append((value_now,next_value.view(-1,1),env_now))
            if done:

                #             print("done")
                next_value = 0
                sqlt.updateBestOrder(reward.item(),action_this_epi)

            expected_state_action_values = (next_value ) + reward.detach()
            final_state_value = (next_value ) + reward.detach()

            if done:
                cnt = 0
                #             for idx in range(t-cnt+1):
                global tree_lstm_memory
                tree_lstm_memory = {}
                DQN.Memory.push(env,expected_state_action_values,final_state_value)
                for pair_s_v in previous_state_list[:0:-1]:
                    cnt += 1
                    if expected_state_action_values > pair_s_v[1]:
                        expected_state_action_values = pair_s_v[1]
                    #                 for idx in range(cnt):
                    expected_state_action_values = expected_state_action_values
                    DQN.Memory.push(pair_s_v[2],expected_state_action_values,final_state_value)
                #                 break
                loss = 0

            if done:
                # break
                loss = DQN.optimize_model()
                # loss = DQN.optimize_model()
                # loss = DQN.optimize_model()
                # loss = DQN.optimize_model()
                losses.append(loss)
                if ((i_episode + 1)%print_every==0):
                    print(np.mean(losses))
                    print("######################Epoch",i_episode//print_every,pg_cost)
                    val_value = DQN.validate(validateSet)
                    print("time",time.time()-startTime)
                    print("~~~~~~~~~~~~~~")
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.cpu().state_dict(), 'CostTraining.pth')
    # policy_net = policy_net.cuda()

if __name__=='__main__':
    sytheticQueries = QueryLoader(QueryDir=config.sytheticDir)
    # print(sytheticQueries)
    JOBQueries = QueryLoader(QueryDir=config.JOBDir)
    Q4,Q1 = k_fold(JOBQueries,10,1)
    # print(Q4,Q1)
    train(Q4+sytheticQueries,Q1)
	


# from PGUtils import PGRunner
# from sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
# from DQN import DQN,ENV
# from TreeLSTM import SPINN
# from JOBParser import DB
import copy
import torch
from torch.nn import init
# from ImportantConfig import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)

featureSize = 128

policy_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
target_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
policy_net.load_state_dict(torch.load("CostTraining.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port,isCostTraining=False,latencyRecord = True,latencyRecordFile = "Latency.json")

DQN = DQN(policy_net,target_net,db_info,pgrunner,device)

def k_fold(input_list,k,ix = 0):
    li = len(input_list)
    kl = (li-1)//k + 1
    train = []
    validate = []
    for idx in range(li):

        if idx%k == ix:
            validate.append(input_list[idx])
        else:
            train.append(input_list[idx])
    return train,validate


def QueryLoader(QueryDir):
    def file_name(file_dir):
        import os
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.sql':
                    L.append(os.path.join(root, file))
        return L
    files = file_name(QueryDir)
    sql_list = []
    for filename in files:
        with open(filename, "r") as f:
            data = f.readlines()
            one_sql = "".join(data)
            sql_list.append(sqlInfo(pgrunner,one_sql,filename))
    return sql_list

def resample_sql(sql_list):
    rewards = []
    reward_sum = 0
    rewardsP = []
    mes = 0
    for sql in sql_list:
        #         sql = val_list[i_episode%len(train_list)]
        pg_cost = sql.getDPlantecy()
        #         continue
        env = ENV(sql,db_info,pgrunner,device)

        for t in count():
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=False)

            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)

            reward, done = env.reward()
            if done:
                mrc = max(np.exp(reward*log(1.5))/pg_cost-1,0)
                rewardsP.append(np.exp(reward*log(1.5)-log(pg_cost)))
                mes += reward*log(1.5)-log(pg_cost)
                rewards.append((mrc,sql))
                reward_sum += mrc
                break
    import random
    print(rewardsP)
    res_sql = []
    print(mes/len(sql_list))
    for idx in range(len(sql_list)):
        rd = random.random()*reward_sum
        for ts in range(len(sql_list)):
            rd -= rewards[ts][0]
            if rd<0:
                res_sql.append(rewards[ts][1])
                break
    return res_sql+sql_list
def train(trainSet,validateSet):

    trainSet_temp = trainSet
    losses = []
    startTime = time.time()
    print_every = 20
    TARGET_UPDATE = 3
    for i_episode in range(0,10000):
        if i_episode % 200 == 100:
            trainSet = resample_sql(trainSet_temp)
        #     sql = random.sample(train_list_back,1)[0][0]
        sqlt = random.sample(trainSet[0:],1)[0]
        pg_cost = sqlt.getDPlantecy()
        env = ENV(sqlt,db_info,pgrunner,device)

        previous_state_list = []
        action_this_epi = []
        nr = True
        nr = random.random()>0.3 or sqlt.getBestOrder()==None
        acBest = (not nr) and random.random()>0.7
        for t in count():
            # beginTime = time.time();
            action_list, chosen_action,all_action = DQN.select_action(env,need_random=nr)
            value_now = env.selectValue(policy_net)
            next_value = torch.min(action_list).detach()
            # e1Time = time.time()
            env_now = copy.deepcopy(env)
            # endTime = time.time()
            # print("make",endTime-startTime,endTime-e1Time)
            if acBest:
                chosen_action = sqlt.getBestOrder()[t]
            left = chosen_action[0]
            right = chosen_action[1]
            env.takeAction(left,right)
            action_this_epi.append((left,right))

            reward, done = env.reward()
            reward = torch.tensor([reward], device=device, dtype = torch.float32).view(-1,1)

            previous_state_list.append((value_now,next_value.view(-1,1),env_now))
            if done:

                #             print("done")
                next_value = 0
                sqlt.updateBestOrder(reward.item(),action_this_epi)

            expected_state_action_values = (next_value ) + reward.detach()
            final_state_value = (next_value ) + reward.detach()

            if done:
                cnt = 0
                #             for idx in range(t-cnt+1):
                global tree_lstm_memory
                tree_lstm_memory = {}
                DQN.Memory.push(env,expected_state_action_values,final_state_value)
                for pair_s_v in previous_state_list[:0:-1]:
                    cnt += 1
                    if expected_state_action_values > pair_s_v[1]:
                        expected_state_action_values = pair_s_v[1]
                    #                 for idx in range(cnt):
                    expected_state_action_values = expected_state_action_values
                    DQN.Memory.push(pair_s_v[2],expected_state_action_values,final_state_value)
                #                 break
                loss = 0

            if done:
                # break
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                loss = DQN.optimize_model()
                losses.append(loss)
                if ((i_episode + 1)%print_every==0):
                    print(np.mean(losses))
                    print("######################Epoch",i_episode//print_every,pg_cost)
                    val_value = DQN.validate(validateSet)
                    print("time",time.time()-startTime)
                    print("~~~~~~~~~~~~~~")
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.cpu().state_dict(), 'LatencyTuning.pth')

if __name__=='__main__':
    sytheticQueries = QueryLoader(QueryDir=config.sytheticDir)
    # print(sytheticQueries)
    JOBQueries = QueryLoader(QueryDir=config.JOBDir)
    Q4,Q1 = k_fold(JOBQueries,10,1)
    # print(Q4,Q1)
    train(Q4+sytheticQueries,Q1)	