class Sample:
    def __init__(self,
                 topology_name=None,
                 frequency=None,
                 attackers_type=None,
                 n_attackers=None,
                 sim_id=None,
                 scenario=None,
                 time=None,
                 attack_is_on=None,
                 routers_feat=None,
                 graph=None
                 ):
        self.topology_name = topology_name
        self.frequency = frequency
        self.attackers_type = attackers_type
        self.n_attackers = n_attackers
        self.sim_id = sim_id
        self.scenario = scenario
        self.time = time
        self.attack_is_on = attack_is_on
        self.routers_feat = routers_feat
        self.graph = graph

    def insert_graph(self, graph):
        self.graph = graph

    def insert_routers_feat(self, routers_feat):
        self.routers_feat = routers_feat

    def insert_topology_name(self, topology_name):
        self.topology_name = topology_name

    def insert_frequency(self, frequency):
        self.frequency = frequency

    def insert_attackers_type(self, attackers_type):
        self.attackers_type = attackers_type

    def insert_n_attackers(self, n_attackers):
        self.n_attackers = n_attackers

    def insert_sim_id(self, sim_id):
        self.sim_id = sim_id

    def insert_scenario(self, scenario):
        self.scenario = scenario

    def insert_time(self, time):
        self.time = time

    def insert_label(self, label):
        self.attack_is_on = label

    def get_sim_setup(self):
        return {'topology_name': self.topology_name,
                'scenario': self.scenario,
                'frequency': self.frequency,
                'attackers_type': self.attackers_type,
                'n_attackers': self.n_attackers,
                'sim_id': self.sim_id}

    def get_sim_setup_key(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.scenario, self.topology_name, self.frequency,
                                          self.attackers_type, self.n_attackers, self.sim_id)

    def get_routers_and_interfaces(self):
        return {rout: [interface for interface in list(rout_dict.keys())] for rout, rout_dict in self.routers_feat.items()}

    def get_routers_feat(self):
        return self.routers_feat

    def get_label(self):
        return self.attack_is_on

    def get_time(self):
        return self.time

    def check_same_sim_id(self, other):
        this_sim_setup = self.get_sim_setup()
        other_sim_setup = other.get_sim_setup()
        for key, value in this_sim_setup.items():
            if not value == other_sim_setup[key]:
                return False
        return True

    def is_next(self, other):
        # Return true if other is the next sample in the same simulation
        if not self.check_same_sim_id(other):
            return False
        else:
            if self.get_time() == other.get_time() - 1:
                return True
            else:
                return False
