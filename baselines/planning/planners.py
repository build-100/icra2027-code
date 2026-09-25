"""Coordinated coverage mechanisms and CGE's two-stage planning adaptation.

No territorial exclusion and no pre-known inter-robot transformations. Every
accepted loop insertion is scored by the same representative Schur objective.
"""
import itertools
import time
import numpy as np
import networkx as nx


class PlanningBudget(Exception): pass


class CoordinatedPlanner:
    def __init__(self,env,method,budget=30000,seed=906300):
        if method == 'cge_dgre_order' and type(self).improve is CoordinatedPlanner.improve:
            raise ValueError('Use CGEDoubleGreedyPlanner for cge_dgre_order')
        self.env,self.method,self.budget=env,method,budget
        self.graph=env.prior.prior_graph
        self.paths=dict(nx.all_pairs_dijkstra_path(self.graph,weight='weight'))
        self.dist=dict(nx.all_pairs_dijkstra_path_length(self.graph,weight='weight'))
        self.used=0; self.candidates=0; self.best=None; self.last=None; self.history=[]
        self.seed=seed

    def step(self,actions):
        if self.used>=self.budget:raise PlanningBudget()
        result=self.env.step(actions);self.used+=1
        return result

    def padding(self,node):
        return min(self.graph.neighbors(node),key=lambda v:(self.graph.edges[node,v]['weight'],v))

    def evaluate(self,routes):
        e=self.env;e.reset();actual=[[s] for s in e.start_labels];distance=0.;reward=0.
        for t in range(min(e.t_max,max(map(len,routes))-1)):
            if e.done:break
            destinations=[route[t+1] if t+1<len(route) else self.padding(actual[i][-1]) for i,route in enumerate(routes)]
            result=self.step([e.prior.action_for_neighbor(actual[i][-1],v) for i,v in enumerate(destinations)])
            for i,v in enumerate(destinations):actual[i].append(v)
            distance+=result.travel_distance;reward+=result.reward
        row={'routes':actual,'success':e.success,'coverage':e.coverage_count(),'c':e.component_count(),
             'T':e.state.time,'S':e.structural_score(),'D':distance,'J':e.structural_score()-e.beta*distance,
             'return':reward,'simulation_steps':self.used}
        self.candidates+=1;self.last=row
        if row['success'] and (self.best is None or row['J']>self.best['J']+1e-12):self.best=row
        self.history.append({'candidate':self.candidates,'simulation_steps':self.used,'success':row['success'],
                             'J':row['J'] if row['success'] else None,'best_J':self.best['J'] if self.best else None})
        return row

    def coverage(self,order):
        """Global unvisited targets, cost-based allocation, no forbidden overlaps."""
        e=self.env;e.reset();routes=[[s] for s in e.start_labels];targets=[None]*e.num_robots
        while e.coverage_count()<e.num_nodes and not e.done:
            visited=set().union(*(set(r) for r in routes));reserved=set()
            for i in order:
                if targets[i] in visited:targets[i]=None
                if targets[i] is not None:reserved.add(targets[i])
            for i in order:
                if targets[i] is None:
                    available=set(e.node_labels)-visited-reserved
                    if available:
                        targets[i]=min(available,key=lambda v:(self.dist[routes[i][-1]][v],v))
                        reserved.add(targets[i])
            destinations=[]
            for i,route in enumerate(routes):
                target=targets[i]
                if target is None:
                    destinations.append(self.padding(route[-1]))
                else:destinations.append(self.paths[route[-1]][target][1])
            self.step([e.prior.action_for_neighbor(routes[i][-1],v) for i,v in enumerate(destinations)])
            for i,v in enumerate(destinations):routes[i].append(v)
        return routes

    def fuse(self,routes):
        """Cost/makespan-ranked sequential links to visited nodes of other components.

        All robots move each joint step; ordinary padding is charged. A new
        alignment becomes usable only after its physical co-visit occurs.
        """
        base=self.evaluate(routes)
        if base['success'] or base['coverage']<self.env.num_nodes:return base
        current=base
        for _ in range(self.env.num_robots-1):
            self.evaluate(current['routes'])
            if self.env.done:return current
            component={node:k for k,c in enumerate(nx.connected_components(self.env.pose_graph)) for node in c}
            candidates=[]
            for i in range(self.env.num_robots):
                ci=component[(i,int(self.env.state.positions[i]))]
                for j in range(self.env.num_robots):
                    if i==j:continue
                    cj=component[(j,int(self.env.state.positions[j]))]
                    if ci==cj:continue
                    for label in set(current['routes'][j]):
                        path=self.paths[current['routes'][i][-1]][label]
                        if len(path)>1:
                            candidates.append((len(path)-1,self.dist[path[0]][label],i,label,path))
            if not candidates:return current
            choices=[]
            # Test alternative shortest component connections instead of forcing a common hub.
            for _,_,i,_,path in sorted(candidates)[:12]:
                trial=[list(r) for r in current['routes']]
                trial[i].extend(path[1:]);length=len(trial[i])
                for j in range(len(trial)):
                    while len(trial[j])<length:trial[j].append(self.padding(trial[j][-1]))
                row=self.evaluate(trial)
                choices.append(row)
            current=min(choices,key=lambda r:(r['c'],-int(r['success']),r['T'],r['D']))
            if current['success'] or current['T']>=self.env.t_max:return current
        return current

    def improve(self,base):
        """CGE-style informative loop insertion; no submodularity guarantee is inherited."""
        current=base
        for _ in range(8):
            winner=current
            seen=set()
            for t in range(current['T']):
                for i,route in enumerate(current['routes']):
                    anchor=route[t]
                    for neighbor in self.graph.neighbors(anchor):
                        identity=(i,t,neighbor)
                        if identity in seen:continue
                        seen.add(identity)
                        # Both intra and inter opportunities are allowed; full replay
                        # determines factor uniqueness and actual S gain.
                        trial=[]
                        for j,r in enumerate(current['routes']):
                            v=neighbor if i==j else self.padding(r[t])
                            trial.append(r[:t+1]+[v,r[t]]+r[t+1:])
                        candidate=self.evaluate(trial)
                        if candidate['success'] and candidate['J']>winner['J']+1e-12:winner=candidate
            if winner is current:break
            current=winner
        return current

    def run(self):
        began=time.perf_counter()
        try:
            # All deterministic priority orders (at most 24), fixed before seeing scores.
            # Give nonlearning planners candidate search, not only one weak attempt.
            for order in itertools.permutations(range(self.env.num_robots)):
                routes=self.coverage(order)
                row=self.evaluate(routes)
                if self.method!='coverage_only' and row['coverage']==self.env.num_nodes and not row['success']:
                    row=self.fuse(routes)
                if self.method in ('cge_adapted','cge_dgre_order') and row['success']:
                    self.improve(row)
        except PlanningBudget:
            pass
        return {'method':self.method,'success':self.best is not None,'selected':self.best,
                'last_candidate':self.last,'simulator_steps':self.used,'candidates':self.candidates,
                'history':self.history,'wall_seconds':time.perf_counter()-began,
                'budget':self.budget,'selection':'best feasible J among generated candidates',
                'deterministic':self.method!='cge_dgre_order','reference_adaptation':'see REFERENCES.md'}
