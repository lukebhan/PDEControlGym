import unittest
from unittest import mock
from ..envs.figure import PlotGenerator
import numpy as np


class TestFigureObjClass(unittest.TestCase):
    def setUp(self):
        T = 4
        dt = 1e-2
        X = 1
        dx = 1e-2
        nt = int(round(T / dt))
        nx = int(round(X / dx))
        self.x = np.linspace(0, 1, nx)
        self.t = np.linspace(0, 4, nt)
        self.fig_settings = {
            "show_error_figure": False,
            "save_fig": False,
            "show_fig": False,
            "fig_xlabel": "x",
            "fig_ylabel": "t",
            "fig_zlabel": "$u(x, t)$",
            "cstride": 50,
            "rstride": 1,
            "fig_size": 516,
        }
        u = []
        for i in range(len(self.t)):
            uArr = []
            for x in range(len(self.x)):
                uArr.append(np.sin(self.t[i]))
            u.append(uArr)
        self.u = np.array(u)

    @mock.patch("pdeGym.envs.figure.plt")
    def test_figure_defaults(self, mock_plt):
        figObj = PlotGenerator()
        figObj.makeFigure(self.x, self.t, self.u)
        assert mock_plt.figure.called
        assert (
            mock_plt.figure().subfigures().__getitem__().subplots().plot.called == False
        )
        assert mock_plt.figure().savefig.called == False
        assert mock_plt.show.called == True

    @mock.patch("pdeGym.envs.figure.plt")
    def test_figure_no_error(self, mock_plt):
        self.fig_settings["save_fig"] = True
        self.fig_settings["file_name"] = "test.png"
        figObj = PlotGenerator(self.fig_settings)
        figObj.makeFigure(self.x, self.t, self.u)
        assert mock_plt.figure.called
        mock_plt.figure().subplots().set_xlabel.assert_called_once_with(
            self.fig_settings["fig_xlabel"]
        )
        mock_plt.figure().subplots().set_zlabel.assert_called_once_with(
            self.fig_settings["fig_zlabel"], rotation=90
        )
        mock_plt.figure().subplots().set_ylabel.assert_called_once_with(
            self.fig_settings["fig_ylabel"]
        )
        assert mock_plt.figure().savefig.called
        assert mock_plt.show.call_count == 0
        assert (
            mock_plt.figure().subfigures().__getitem__().subplots().plot.called == False
        )

    @mock.patch("pdeGym.envs.figure.plt")
    def test_figure_with_error(self, mock_plt):
        self.fig_settings["show_fig"] = True
        self.fig_settings["show_error_figure"] = True
        self.fig_settings["error_fig_xlabel"] = "t"
        self.fig_settings["error_fig_ylabel"] = "Error"
        figObj = PlotGenerator(self.fig_settings)
        figObj.makeFigure(self.x, self.t, self.u)
        assert mock_plt.figure.called
        assert (
            mock_plt.figure().subfigures().__getitem__().subplots().plot_surface.called
        )
        assert mock_plt.figure().subfigures().__getitem__().subplots().plot.called


if __name__ == "__main__":
    unittest.main()
