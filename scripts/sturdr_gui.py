import sys
import os
from threading import Thread, Condition
from datetime import datetime, timedelta
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QTableView,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QHeaderView,
    QGraphicsSimpleTextItem,
)
from PyQt6.QtCharts import (
    QChart,
    QPolarChart,
    QBarSeries,
    QBarSet,
    QBarCategoryAxis,
    QScatterSeries,
    QValueAxis,
    QChartView,
)
from PyQt6.QtGui import QFont, QPen, QColor, QPainter
from PyQt6.QtCore import (
    Qt,
    QObject,
    QAbstractTableModel,
    QModelIndex,
    QPointF,
    QMargins,
    QTimer,
    pyqtSignal,
)
from utils.plotters import FoliumWidget

dir = os.path.realpath(os.path.dirname(__file__))
print(dir)
sys.path.append("/usr/local/lib/python3/dist-packages/")
sys.path.append(dir + "/../build/src/sturdds/rtps/sturdds/")

from fastdds import (
    DomainParticipantFactory,
    DomainParticipantQos,
    TypeSupport,
    TopicQos,
    SubscriberQos,
    DataReaderQos,
    DataReaderListener,
    HistoryQosPolicy,
    KEEP_ALL_HISTORY_QOS,
    SampleInfo,
    RETCODE_OK,
)
from NavMessage import NavMessage, NavMessagePubSubType
from ChannelMessage import ChannelMessage, ChannelMessagePubSubType

MYGREEN = QColor(8, 171, 92)
MYBLUE = QColor(30, 159, 222)
MYWHITE = QColor(228, 230, 235)
MYLIGHTGREY = QColor(149, 160, 176)
MYDARKGREY = QColor(42, 42, 42)


def gps_to_utc_time(week, tow):
    # GPS Epoch: January 6, 1980
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Calculate the total seconds since the GPS epoch
    total_seconds = week * 604800 + tow

    # convert to utc time (-18 leap seconds)
    # return gps_epoch + timedelta(seconds=total_seconds - 18)
    return gps_epoch + timedelta(seconds=total_seconds)


def calc_dop(az: np.ndarray[float], el: np.ndarray[float]):
    if np.any(np.isnan(az)) or np.any(np.isnan(el)):
        return "---", "---", "---", "---", "---"

    # print(f"az = {az}")
    # print(f"el = {el}")
    az_ = np.deg2rad(az)
    el_ = np.rad2deg(el)
    saz = np.sin(az_)
    caz = np.cos(az_)
    sel = np.sin(el_)
    cel = np.cos(el_)
    H = np.column_stack((saz * cel, caz * cel, sel, np.ones(az.size)))
    # print(f"H = \n{H}")
    DOP = np.diag(np.linalg.inv(H.T @ H))
    # print(f"DOP = {DOP}")
    gdop = np.linalg.norm(DOP)
    pdop = np.linalg.norm(DOP[:3])
    hdop = np.linalg.norm(DOP[:2])
    vdop = DOP[2]
    tdop = DOP[3]
    return gdop, pdop, hdop, vdop, tdop


# Signal to communicate data from the DDS navigation message to the GUI thread
class NavSignal(QObject):
    new_data = pyqtSignal(str, float, float, float, float, float)


# Signal to communicate data from the DDS channel messages to the GUI thread
class ChannelSignal(QObject):
    new_data = pyqtSignal(int, int, float, float, float)


# Navigation data Listener
class NavListener(DataReaderListener):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def on_data_available(self, reader):
        info = SampleInfo()
        msg = NavMessage()
        if reader.take_next_sample(msg, info) == RETCODE_OK:
            self._signal.new_data.emit(
                gps_to_utc_time(msg.Week(), msg.ToW()).strftime("%m/%d/%Y\n%H:%M:%S"),
                msg.Lat(),
                msg.Lon(),
                msg.H(),
                np.linalg.norm([msg.Vn(), msg.Ve(), msg.Vd()]),
                msg.Yaw() % 360,
            )

    def on_subscription_matched(self, reader, info):
        if info.current_count_change == 1:
            print(f"Navigation subscriber matched a publisher (Reader: {reader.guid()})")
        elif info.current_count_change == -1:
            print(f"Navigation subscriber unmatched a publisher (Reader: {reader.guid()})")


# Channels data Listener
class ChannelListener(DataReaderListener):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def on_data_available(self, reader):
        info = SampleInfo()
        msg = ChannelMessage()
        if reader.take_next_sample(msg, info) == RETCODE_OK:
            # print(
            #     f"New channel data: ChID = {msg.ChannelID()}, SvID = {msg.SatelliteID()}, CNo = "
            #     f"{msg.CNo():.2f}, Az = {msg.Azimuth():.2f}, El = {msg.Elevation():.2f}"
            # )
            self._signal.new_data.emit(
                msg.ChannelID(),
                msg.SatelliteID(),
                msg.CNo(),
                msg.Azimuth() % 360,
                msg.Elevation(),
            )

    def on_subscription_matched(self, reader, info):
        if info.current_count_change == 1:
            print(f"Channels subscriber matched a publisher (Reader: {reader.guid()})")
        elif info.current_count_change == -1:
            print(f"Channels subscriber unmatched a publisher (Reader: {reader.guid()})")


# Custom Data Table Widget
class SturdrTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = {
            "UTC Time": "---",
            "Latitude [\N{DEGREE SIGN}]": "---",
            "Longitude [\N{DEGREE SIGN}]": "---",
            "Height [m]": "---",
            "Speed [m/s]": "---",
            "Heading [\N{DEGREE SIGN}]": "---",
            "PDOP": "---",
            "HDOP": "---",
            "SV Tracked": "---",
        }
        self._keys = list(self._data.keys())

    def rowCount(self, parent=QModelIndex()):
        return 9  # len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        key = self._keys[row]
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return key
            elif col == 1:
                return self._data[key]
        elif role == Qt.ItemDataRole.ForegroundRole:
            return MYWHITE
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if section == 0:
                    return "Key"
                elif section == 1:
                    return "Value"
        return None

    def Update1(
        self,
        utc_time="---",
        lat="---",
        lon="---",
        h="---",
        speed="---",
        heading="---",
    ):
        self._data["UTC Time"] = utc_time
        self._data["Latitude [\N{DEGREE SIGN}]"] = f"{lat:.8f}" if isinstance(lat, float) else lat
        self._data["Longitude [\N{DEGREE SIGN}]"] = f"{lon:.8f}" if isinstance(lon, float) else lon
        self._data["Height [m]"] = f"{h:.1f}" if isinstance(h, float) else h
        self._data["Speed [m/s]"] = f"{speed:.2f}" if isinstance(speed, float) else speed
        self._data["Heading [\N{DEGREE SIGN}]"] = (
            f"{heading:.1f}" if isinstance(heading, float) else heading
        )
        # print(self._data)
        self.dataChanged.emit(
            self.index(0, 1), self.index(self.rowCount() - 1, 1), [Qt.ItemDataRole.DisplayRole]
        )

    def Update2(
        self,
        pdop="---",
        hdop="---",
        n_sv="---",
    ):
        self._data["PDOP"] = f"{pdop:.3f}" if isinstance(pdop, float) else pdop
        self._data["HDOP"] = f"{hdop:.3f}" if isinstance(hdop, float) else hdop
        self._data["SV Tracked"] = f"{n_sv:d}" if isinstance(n_sv, float) else n_sv
        # print(self._data)
        self.dataChanged.emit(
            self.index(0, 1), self.index(self.rowCount() - 1, 1), [Qt.ItemDataRole.DisplayRole]
        )


# A simple skyplot
class SkyplotWidget(QChartView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._chart = QPolarChart()
        self.setChart(self._chart)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chart.setMargins(QMargins(10, 5, 10, 5))
        self._chart.legend().setVisible(False)
        self._chart.setBackgroundBrush(MYDARKGREY)
        self._pen = QPen(MYLIGHTGREY)
        self._font = QFont()
        self._font.setPointSize(10)

        self._points = []
        self._point_labels = []
        self._scatter_series = QScatterSeries()
        self._scatter_series.setMarkerSize(16)
        self._scatter_series.setColor(MYGREEN)
        self._scatter_series.setBorderColor(MYGREEN)
        self._chart.addSeries(self._scatter_series)

        # Angular axis (azimuth)
        self._az_axis = QValueAxis()
        self._az_axis.setLabelFormat("%d")
        self._az_axis.setRange(0, 360)
        self._az_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self._az_axis.setTickCount(13)
        self._az_axis.setLabelsVisible(False)
        self._az_axis.setLinePen(self._pen)
        self._az_axis.setGridLinePen(self._pen)
        self._chart.addAxis(self._az_axis, QPolarChart.PolarOrientation.PolarOrientationAngular)

        # Radial axis (elevation - needs scaling)
        self._el_axis = QValueAxis()
        self._el_axis.setRange(0, 90)  # Elevation from 0 to 90 degrees
        self._el_axis.setLabelFormat("%d")
        self._el_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self._el_axis.setTickCount(4)
        self._el_axis.setLabelsVisible(False)
        self._el_axis.setLinePen(self._pen)
        self._el_axis.setGridLinePen(self._pen)
        self._chart.addAxis(self._el_axis, QPolarChart.PolarOrientation.PolarOrientationRadial)

        # Connect series to axes
        self._scatter_series.attachAxis(self._az_axis)
        self._scatter_series.attachAxis(self._el_axis)

        # custom axis labels
        self._angular_labels = []
        self._radial_labels = []
        self._create_angular_labels()
        self._create_radial_labels()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._create_angular_labels()
        self._create_radial_labels()
        [self._create_point_label(pt, str(i), i) for i, pt in enumerate(self._points)]

    def update_plot(self, new_az: float, new_el: float, svid: int, index: int):
        # print(f"{index}: sv = {svid}, az = {new_az:.2f}, el = {new_el:.2f}")
        if np.isnan(new_az) or np.isnan(new_el):
            return
        new_point = QPointF(new_az, 90 - new_el)
        if not self._points or index >= len(self._points):
            self._scatter_series.append(new_point)
            self._points.append(new_point)
            self._create_point_label(new_point, str(svid))
        else:
            self._scatter_series.replace(self._points[index], new_point)
            self._points[index] = new_point
            if self._point_labels:
                self._update_point_label(new_point, str(svid), index)
        # self.chart().update()  # trigger a redraw

    # def _redraw(self):
    #     self.chart().update()

    def _create_angular_labels(self):
        scene = self._chart.scene()
        angular_labels = {
            0: "N",
            30: "30\N{DEGREE SIGN}",
            60: "60\N{DEGREE SIGN}",
            90: "E",
            120: "120\N{DEGREE SIGN}",
            150: "150\N{DEGREE SIGN}",
            180: "S",
            210: "210\N{DEGREE SIGN}",
            240: "240\N{DEGREE SIGN}",
            270: "W",
            300: "300\N{DEGREE SIGN}",
            330: "330\N{DEGREE SIGN}",
        }

        # Remove existing labels
        for item in self._angular_labels:
            scene.removeItem(item)
        self._angular_labels = []

        for angle, label_text in angular_labels.items():
            polar_point = QPointF(angle, self._el_axis.max() * 1.1)
            point = self._chart.mapToPosition(polar_point, self._scatter_series)
            text_item = QGraphicsSimpleTextItem(label_text)
            text_item.setFont(self._font)
            text_item.setBrush(MYWHITE)
            text_rect = text_item.boundingRect()
            text_item.setPos(point.x() - text_rect.width() / 2, point.y() - text_rect.height() / 2)
            scene.addItem(text_item)
            self._angular_labels.append(text_item)

    def _create_radial_labels(self):
        scene = self._chart.scene()
        radial_values = range(0, 91, 30)
        angle_for_labels = 270

        # Remove any existing labels
        for item in self._radial_labels:
            scene.removeItem(item)
        self._radial_labels = []

        for ii in range(len(radial_values)):
            polar_point = QPointF(angle_for_labels, radial_values[-(ii + 1)])
            point = self._chart.mapToPosition(polar_point, self._scatter_series)

            text_item = QGraphicsSimpleTextItem(f"{radial_values[ii]}°")
            text_item.setFont(self._font)
            text_item.setBrush(MYWHITE)
            text_rect = text_item.boundingRect()
            text_item.setPos(
                point.x() - text_rect.width() / 2 + 15, point.y() - text_rect.height() / 2 - 7
            )
            scene.addItem(text_item)
            self._radial_labels.append(text_item)

    def _create_point_label(self, point: QPointF, label: str):
        map_pt = self._chart.mapToPosition(point)
        font = QFont()
        font.setPointSize(8)
        item = QGraphicsSimpleTextItem(label)
        item.setFont(font)
        item.setBrush(Qt.GlobalColor.white)
        rect = item.boundingRect()
        item.setPos(map_pt.x() - rect.width() / 2 - 2, map_pt.y() - rect.height() / 2 - 2)
        self._chart.scene().addItem(item)
        self._point_labels.append(item)

    def _update_point_label(self, point: QPointF, label: str, index: int):
        map_pt = self._chart.mapToPosition(point)
        self._point_labels[index].setText(label)
        rect = self._point_labels[index].boundingRect()
        self._point_labels[index].setPos(
            map_pt.x() - rect.width() / 2, map_pt.y() - rect.height() / 2
        )


# A simple bar chart
class BarChartWidget(QChartView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bar_set = QBarSet("")
        self._bar_series = QBarSeries()
        self._axis_x = QBarCategoryAxis()
        self._axis_y = QValueAxis()
        self._chart = QChart()
        self._categories = []
        self._data = []

        # self._chart.setTitle("Simple Chart")
        self._chart.setBackgroundBrush(MYDARKGREY)
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setChart(self._chart)

        font = QFont()
        font.setPointSize(13)
        pen = QPen(MYLIGHTGREY)
        color = MYWHITE

        # build chart
        self._axis_y.setRange(0, 50)
        self._axis_y.setTitleText("C/No [dB-Hz]")
        self._axis_y.setTitleBrush(color)
        self._axis_y.setTitleFont(font)
        self._axis_y.setTickType(QValueAxis.TickType.TicksDynamic)
        self._axis_y.setLabelFormat("%d")
        # self._axis_y.setLabelsFont(font)
        self._axis_y.setTickInterval(10)
        self._axis_y.setLinePen(pen)
        self._axis_y.setGridLinePen(pen)
        self._axis_y.setLabelsBrush(color)
        # self._axis_x.setLabelsFont(font)
        self._axis_x.setLinePen(pen)
        self._axis_x.setGridLinePen(pen)
        self._axis_x.setLabelsBrush(color)
        self._axis_x.setGridLineVisible(False)
        self._bar_set.setColor(MYGREEN)
        self._bar_series.append(self._bar_set)
        self._bar_series.setBarWidth(0.85)
        self._chart.addSeries(self._bar_series)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._bar_series.attachAxis(self._axis_x)
        self._bar_series.attachAxis(self._axis_y)
        self._chart.setContentsMargins(0, 0, 0, 0)
        self._chart.setMargins(QMargins(2, 0, 8, 0))
        self._chart.legend().setVisible(False)

    def update_plot(self, new_cno: float, svid: int, index: int):
        # print(f"{index}: sv = {svid}, cno = {new_cno:.2f}")
        if np.isnan(new_cno):
            return
        new_category = f"G{svid:d}"
        if self._bar_set.count() == 0 or index >= self._bar_set.count():
            # add new point
            self._data.append(new_cno)
            self._bar_set.append(new_cno)
            self._categories.append(new_category)
            self._axis_x.append(new_category)
        else:
            self._data[index] = new_cno
            self._bar_set.replace(index, new_cno)
            self._axis_x.replace(self._categories[index], new_category)
            self._categories[index] = new_category
        # self.chart().update()  # Trigger a redraw

    # def _redraw(self):
    #     self.chart().update()


# Main gui class
class SturdrGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SturDR GUI")
        self.setGeometry(10, 10, 1100, 900)  # Adjust initial geometry
        self.setStyleSheet("background-color: rgb(42,42,42);")
        self._central_layout = QVBoxLayout()
        self._top_layout = QHBoxLayout()
        self._bottom_layout = QHBoxLayout()

        self._az = []
        self._el = []
        self._sv = []
        self._n_sv = 0
        self._cnt = 0
        self._has_nav_sol = False

        # create folium plot
        kwargs = {
            "color": "#ff0000",
            "weight": 5,
            "opacity": 1,
            "geobasemap": "satellite",
            "zoom": 20,
        }
        self._map_widget = FoliumWidget(maxlen=100, **kwargs)
        self._map_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._top_layout.addWidget(self._map_widget, 8)

        # create data table
        self._data_table_model = SturdrTableModel()
        self._data_table = QTableView()
        self._data_table.setModel(self._data_table_model)
        self._data_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data_table.horizontalHeader().setVisible(False)
        self._data_table.verticalHeader().setVisible(False)
        hor_header = self._data_table.horizontalHeader()
        hor_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hor_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        ver_header = self._data_table.verticalHeader()
        ver_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._top_layout.addWidget(self._data_table, 3)

        # create bar chart
        self._barchart_widget = BarChartWidget()
        self._barchart_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._bottom_layout.addWidget(self._barchart_widget, 7)

        # create skyplot
        self._skyplot_widget = SkyplotWidget()
        self._skyplot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._bottom_layout.addWidget(self._skyplot_widget, 3)

        # set the layout of the window
        self._top_container = QWidget()
        self._bottom_container = QWidget()
        self._top_container.setLayout(self._top_layout)
        self._bottom_container.setLayout(self._bottom_layout)
        self._central_layout.addWidget(self._top_container, 3)
        self._central_layout.addWidget(self._bottom_container, 2)
        self.setLayout(self._central_layout)

        self._shutdown_event = Condition()

        # create a dds navigation subscriber
        self._dds_nav_signal = NavSignal()
        self._dds_nav_signal.new_data.connect(self._update_nav)
        self._dds_nav_thread = Thread(target=self._subscriber, args=["sturdr-navigator"])
        self._dds_nav_thread.start()

        # create a dds channels subscriber
        self._dds_channels_signal = ChannelSignal()
        self._dds_channels_signal.new_data.connect(self._update_charts)
        self._dds_channels_thread = Thread(target=self._subscriber, args=["sturdr-channels"])
        self._dds_channels_thread.start()

    def closeEvent(self, event):
        print("GUI closed. Notifying DDS subscriber to stop ...")
        with self._shutdown_event:
            self._shutdown_event.notify_all()
        self._dds_nav_thread.join()  # Wait for the DDS thread to finish
        self._dds_channels_thread.join()
        event.accept()

    def _update_nav(self, utc_time, lat, lon, h, speed, yaw):
        self._map_widget.UpdateMap([lat, lon], yaw)
        self._data_table_model.Update1(utc_time, lat, lon, h, speed, yaw)
        self._has_nav_sol = True

    def _update_charts(self, ch_id, sv_id, cno, az, el):
        self._skyplot_widget.update_plot(az, el, sv_id, ch_id)
        self._barchart_widget.update_plot(cno, sv_id, ch_id)
        if sv_id not in self._sv:
            self._az.append(az)
            self._el.append(el)
            self._sv.append(sv_id)
            self._n_sv += 1
            self._cnt = 0
        else:
            self._az[ch_id] = az
            self._el[ch_id] = el
            self._sv[ch_id] = sv_id
            self._cnt += 1
            if self._cnt == self._n_sv:
                if self._has_nav_sol and self._n_sv > 3:
                    _, pdop, hdop, _, _ = calc_dop(np.asarray(self._az), np.asarray(self._el))
                    self._data_table_model.Update2(pdop, hdop, self._n_sv)
                self._cnt = 0

    # def _redraw(self):
    #     self._skyplot_widget.chart().update()
    #     self._barchart_widget.chart().update()

    def _subscriber(self, topic_name):
        try:
            factory = DomainParticipantFactory.get_instance()

            # init
            if topic_name == "sturdr-navigator":
                topic_data_type = NavMessagePubSubType()
                listener = NavListener(self._dds_nav_signal)
            elif topic_name == "sturdr-channels":
                topic_data_type = ChannelMessagePubSubType()
                listener = ChannelListener(self._dds_channels_signal)

            # create participant
            participant = None
            participant_qos = DomainParticipantQos()
            factory.get_default_participant_qos(participant_qos)
            participant = factory.create_participant(0, participant_qos)
            if participant is None:
                raise Exception("Could not create dds participant ...")

            # create topic
            type_support = TypeSupport(topic_data_type)
            participant.register_type(type_support)

            topic = None
            topic_qos = TopicQos()
            participant.get_default_topic_qos(topic_qos)
            topic = participant.create_topic(topic_name, topic_data_type.get_name(), topic_qos)
            if topic is None:
                raise Exception("Could not create dds topic ...")

            # create subscriber
            subscriber = None
            subscriber_qos = SubscriberQos()
            participant.get_default_subscriber_qos(subscriber_qos)
            subscriber = participant.create_subscriber(subscriber_qos)
            if subscriber is None:
                raise Exception("Could not create dds subscriber ...")

            # create listener
            history_qos = HistoryQosPolicy()
            history_qos.kind = KEEP_ALL_HISTORY_QOS
            reader_qos = DataReaderQos()
            subscriber.get_default_datareader_qos(reader_qos)
            reader_qos.history(history_qos)
            reader = subscriber.create_datareader(topic, reader_qos, listener)

            # wait until notified to stop
            with self._shutdown_event:
                self._shutdown_event.wait()

        except Exception as e:
            print(f"Subscriber Error: {e}")

        finally:
            participant.delete_contained_entities()
            factory.delete_participant(participant)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SturdrGui()
    win.show()
    sys.exit(app.exec())
