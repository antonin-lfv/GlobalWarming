import streamlit as st

from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
import time
import random as rd

st.set_page_config(layout="wide", page_title="My app", menu_items={
    'About': "My app made with love"
})

# entity
EMPTY = 0
TREE = 1
FACTORY = 4
CAR = 2
# CO2
GREEN = 0
YELLOW = 1
ORANGE = 2
RED = 3

c1, c2, c3 = st.columns((1, 0.1, 1))
with c1:
    size = st.slider(label="Size of the matrix", min_value=20, max_value=200, value=50, step=1)
with c3:
    proportion_factories = st.slider(label="Proportion of Factories", min_value=0., max_value=1., value=0.04, step=0.01)
    proportion_cars = st.slider(label="Proportion of Cars", min_value=0., max_value=1., value=0.04, step=0.01)
    proportion_trees = st.slider(label="Proportion of Trees", min_value=0., max_value=1., value=0.1, step=0.01)


class Grid:
    def __init__(self, size=100, proportion_factories=0.04, proportion_cars=0.04, proportion_trees=0.1, tree_power=3):
        """
        On crée la liste des CO2 qui vaut 0 partout
        On crée la liste des entités qui vaut 0 partout (EMPTY)
        On ajoute des entités avec __set_entity()
        On colorie les entités sur la carte CO2 avec __color_CO2_with_entity()
        """
        self.size = size
        self.grid_entity = self.generate_grid()
        self.grid_CO2 = self.generate_grid()
        self.step_CO2 = []
        self.step_entity = []
        self.proportion_factories = proportion_factories
        self.proportion_cars = proportion_cars
        self.proportion_trees = proportion_trees
        self.tree_power = tree_power
        self.__set_entity()
        self.__color_CO2_with_entity()

    def generate_grid(self):
        """
        Init de grille de 0
        """
        g_grid = np.zeros((self.size, self.size))
        return g_grid

    def __set_entity(self):
        """
        AJouter des entités
        """
        for i in range(self.size):
            for j in range(self.size):
                p = rd.randint(0, 100)
                if p <= self.proportion_factories * 100:
                    self.grid_entity[i, j] = FACTORY
                elif p <= self.proportion_factories * 100 + self.proportion_cars * 100:
                    self.grid_entity[i, j] = CAR
                elif p <= self.proportion_factories * 100 + self.proportion_cars * 100 + self.proportion_trees * 100:
                    self.grid_entity[i, j] = TREE
                else:
                    ...

    def __color_CO2_with_entity(self):
        """
        Coloration des entités sur la carte CO2
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid_entity[i, j] == FACTORY:
                    self.grid_CO2[i, j] = RED
                elif self.grid_entity[i, j] == CAR:
                    self.grid_CO2[i, j] = ORANGE
        self.step_CO2.append(self.grid_CO2.copy())
        self.step_entity.append(self.grid_entity.copy())

    def __entity_to_str(self, step):
        entity_str = []
        for row in self.step_entity[step]:
            to_add = []
            for elem in row:
                if elem == EMPTY:
                    to_add.append(' ')
                elif elem == TREE:
                    to_add.append('tree')
                elif elem == FACTORY:
                    to_add.append('factory')
                else:
                    to_add.append('car')
            entity_str.append(to_add)
        return entity_str

    def heatmap_CO2(self, step=-1, display=True, show_entity=False, get_heatmap=False):
        """
        :param show_entity: display entity on figure
        :param step: numéro de l'étape à afficher
        :param display: affichage ou pas
        :param grille: list of lists
        :return: heatmap
        """
        colorscale = [[0., '#2B812B'], [0.33, '#E9E439'],
                      [0.66, 'orange'], [1., '#E52216']]

        if get_heatmap:
            fig = go.Heatmap(z=self.step_CO2[step], colorscale=colorscale)
            return fig

        if show_entity:
            fig = go.Figure(data=go.Heatmap(z=self.step_CO2[step], colorscale=colorscale,
                                            text=self.__entity_to_str(step),
                                            texttemplate="%{text}",
                                            textfont={"size": 2}
                                            ))
        else:
            fig = go.Figure(data=go.Heatmap(z=self.step_CO2[step], colorscale=colorscale,
                                            ))

        step_title = f"step {step + 1}/{len(self.step_CO2) - 1}" if step != -1 else f"step {len(self.step_CO2) + 1}/{len(self.step_CO2) - 1}"
        fig.update_layout(title=step_title +
                                f"  | Proportion of Factory: <b>{round(self.proportion_factories * 100, 3)}%</b>"
                                f" Proportion of Car: <b>{round(self.proportion_cars * 100, 3)}%</b>"
                                f" Proportion of Tree: <b>{round(self.proportion_trees * 100, 3)}%</b>")
        if display:
            plot(fig)
        return fig

    def is_in_grid(self):
        ...

    def move_cars(self):
        index_i, index_j = np.where(self.grid_entity == CAR)
        for i, j in zip(index_i, index_j):
            ...

    def neighbors_cell_change(self, i: int, j: int, new_grid):
        # crée une liste avec les voisins de la cellule en position i j (en gérant les bords)
        if i == 0:
            if j == 0:
                neighbors = [new_grid[i, j + 1], new_grid[i + 1, j], new_grid[i + 1, j + 1]]
                trees_nb = [self.grid_entity[i, j + 1], self.grid_entity[i + 1, j],
                            self.grid_entity[i + 1, j + 1]].count(1)
            if j == len(new_grid) - 1:
                neighbors = [new_grid[i + 1, j], new_grid[i + 1, j - 1], new_grid[i, j - 1]]
                trees_nb = [self.grid_entity[i + 1, j], self.grid_entity[i + 1, j - 1],
                            self.grid_entity[i, j - 1]].count(1)
            else:
                neighbors = [new_grid[i, j + 1], new_grid[i + 1, j], new_grid[i + 1, j + 1],
                             new_grid[i + 1, j - 1], new_grid[i, j - 1]]
                trees_nb = [self.grid_entity[i, j + 1], self.grid_entity[i + 1, j], self.grid_entity[i + 1, j + 1],
                            self.grid_entity[i + 1, j - 1], self.grid_entity[i, j - 1]].count(1)
        elif i == len(new_grid) - 1:
            if j == 0:
                neighbors = [new_grid[i, j + 1], new_grid[i - 1, j + 1], new_grid[i - 1, j]]
                trees_nb = [self.grid_entity[i, j + 1], self.grid_entity[i - 1, j + 1],
                            self.grid_entity[i - 1, j]].count(1)
            elif j == len(new_grid) - 1:
                neighbors = [new_grid[i - 1, j - 1],
                             new_grid[i, j - 1], new_grid[i - 1, j]]
                trees_nb = [self.grid_entity[i - 1, j - 1],
                            self.grid_entity[i, j - 1], self.grid_entity[i - 1, j]].count(1)
            else:
                neighbors = [new_grid[i, j + 1], new_grid[i - 1, j + 1], new_grid[i - 1, j - 1],
                             new_grid[i, j - 1], new_grid[i - 1, j]]
                trees_nb = [self.grid_entity[i, j + 1], self.grid_entity[i - 1, j + 1], self.grid_entity[i - 1, j - 1],
                            self.grid_entity[i, j - 1], self.grid_entity[i - 1, j]].count(1)
        else:
            if j == 0:
                neighbors = [new_grid[i, j + 1], new_grid[i + 1, j], new_grid[i + 1, j + 1],
                             new_grid[i - 1, j + 1], new_grid[i - 1, j]]
                trees_nb = [self.grid_entity[i, j + 1], self.grid_entity[i + 1, j], self.grid_entity[i + 1, j + 1],
                            self.grid_entity[i - 1, j + 1], self.grid_entity[i - 1, j]].count(1)
            elif j == len(new_grid) - 1:
                neighbors = [new_grid[i + 1, j], new_grid[i + 1, j - 1], new_grid[i - 1, j - 1],
                             new_grid[i, j - 1], new_grid[i - 1, j]]
                trees_nb = [self.grid_entity[i + 1, j], self.grid_entity[i + 1, j - 1], self.grid_entity[i - 1, j - 1],
                            self.grid_entity[i, j - 1], self.grid_entity[i - 1, j]].count(1)
            else:
                neighbors = [
                    new_grid[i, j + 1], new_grid[i + 1, j],
                    new_grid[i + 1, j + 1], new_grid[i + 1, j - 1],
                    new_grid[i - 1, j + 1], new_grid[i - 1, j - 1],
                    new_grid[i, j - 1], new_grid[i - 1, j]
                ]
                trees_nb = [
                    self.grid_entity[i, j + 1], self.grid_entity[i + 1, j],
                    self.grid_entity[i + 1, j + 1], self.grid_entity[i + 1, j - 1],
                    self.grid_entity[i - 1, j + 1], self.grid_entity[i - 1, j - 1],
                    self.grid_entity[i, j - 1], self.grid_entity[i - 1, j]
                ].count(1)
        # faire la moyenne des valeurs de la liste en rendant les arbres inhibiteurs
        somme = sum(neighbors) - self.tree_power * trees_nb
        # si la moyenne est inférieure à 1, la cellule est verte
        if somme <= 1:
            return GREEN
        elif 8 >= somme >= 2:
            return YELLOW
        elif 12 >= somme >= 9:
            return ORANGE
        elif somme >= 13:
            return RED

    def evolution(self, steps_amount: int):
        for _ in range(steps_amount):
            new_grid = self.step_CO2[-1].copy()
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid_entity[i, j] == EMPTY:
                        self.grid_CO2[i, j] = self.neighbors_cell_change(i, j, new_grid)
            self.step_CO2.append(self.grid_CO2.copy())
            self.step_entity.append(self.grid_entity.copy())

    def launch_simulation(self, steps_amount: int = 6):
        self.evolution(steps_amount=steps_amount)
        # Create figure
        fig = go.Figure()
        # Add traces, one for each slider step
        for step in range(steps_amount):
            fig.add_trace(
                self.heatmap_CO2(step=step, get_heatmap=True))
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {
                          "title": f"Slider switched to step: {i + 1}/{len(fig.data)}"
                                   f"  | Proportion of Factory: <b>{round(self.proportion_factories * 100, 3)}%</b>"
                                   f" Proportion of Car: <b>{round(self.proportion_cars * 100, 3)}%</b>"
                                   f" Proportion of Tree: <b>{round(self.proportion_trees * 100, 3)}%</b>"}]
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        fig.data[0].visible = True
        sliders = [dict(
            active=1,
            currentvalue={"prefix": "Step: "},
            pad={"t": len(fig.data)},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            width=800,
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True)


g = Grid(size=size,
         proportion_cars=proportion_cars,
         proportion_trees=proportion_trees,
         proportion_factories=proportion_factories)

g.launch_simulation(steps_amount=30)
