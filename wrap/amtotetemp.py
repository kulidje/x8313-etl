# This module wraps the AmTote webservices for placing bets

from pandas import DataFrame
from suds import client
import requests

# IP for the web services description language (WSDL), which is only accessible after connecting the the VPN
# TODO: recommendation for new naming convention
# Use 3 letter codes for account names: HAM - Gerald Hamm, X8C - X8 Canadian
# Recommend: WSDL_DEV, WSDL_HAM, WSDL_X8C
# TODO: It doesn't look like we actually have WSDL
#WSDL_DEV = 'file:///c:/data/projects/x8313/horse/FileBet.wsdl'
WSDL_DEV = 'http://10.1.0.39:8000/WWFileBet?wsdl'
WSDL_PROD = 'http://10.15.200.190:8000/eot/filebet/ww800679740?singleWsdl'
WSDL_PROD_X8 = 'http://10.15.200.190:8000/eot/filebet/bi800421505?singleWsdl'

# Current public static IP for production server
STATIC_PROD_IP = '18.217.169.51'

# TODO: Do we need to define an Exception type?

class FileBetTemp:
    """ Wrapper for AmTote FileBet web services for automatically placing bets

    Attributes:
        url:
        bet_string:
        client:
        status:
        balance:

    Notes:
       filebet = FileBet() # connect to the service and potentially specify the url
       response = filebet.get_programs() # get a df of available programs
       response = filebet.submit_bets(df_bet) # submit bets using a DataFrame of bet info
       filebet.close() # close the channel post placing the bets
    """

    def __init__(self, url=WSDL_DEV):
        """ Initialization Function

        Connects to the web services and raise an exception if unable to connect.

        Args:
            url: The WSDL constant the defines the connection.
                WSDL_DEV - Development connection
                WSDL_PROD - Gereld Hamm account
                WSDL_PROD_X8 - X8 Canadian Corporate account

        Raises:
            Exception

        """

        # Initialize Attributes
        self.url = url
        self.bet_string = ''
        self.client = None
        self.status = None
        self.balance = None

        print('Using WSDL:' + self.url)

        # Raise exception if we are using a production account and the public
        # IP of the server does not match our production static IP.

        if url != WSDL_DEV:
            currentIP = requests.get("http://ipecho.net/plain?").text
            if  currentIP != STATIC_PROD_IP:
                raise Exception(
                    "FileBet __init__ exception: current IP ("
                    + currentIP
                    + ") does not match production IP ("
                    + STATIC_PROD_IP
                    + ")")

        # Connect client

        try:
            self.client = client.Client(self.url)
        except:
            raise Exception('FileBet ERROR: unable to connect to FileBet client - check VPN health.')

        print('WSDL created')

        # TODO: remove, replaced with above
        # check, if using WSDL_PROD, that you are on Production EC2 Instance
        #if url == WSDL_PROD:
        #    if requests.get("http://ipecho.net/plain?").text != '18.217.169.51':
        #        raise Exception('FileBet ERROR: not using Production Instance with AmTote VPN.')
        #try:
        #    self.client = client.Client(self.url)
        #except:
        #    raise Exception('FileBet ERROR: unable to connect to FileBet client - check VPN health.')

        # call CheckStatus() to verify if AmTote system is available
        self.check_status()

        return

        # response '0/120' means 0 channels open / 120 channels available - this should only be executed at SOD (Start of Day)
        if self.status == '0/120':
            print('Initializing FileBet Client..')

            # SOD Best Practices:
            self.client.service.ResetChannels()

            # call Initialize()
            response = self.client.service.Initialize()
            print('FileBet client.Initialize() response = %s' % response)
            if int(response) == 0:  # this means it's broken on amtote end
                raise Exception('FileBet client.Initialize() response = %s Gateway needs to be reset - email midatlantic_hub@amtote.com' % response)

            # call OpenChannel() before placing bets
            self.balance = float(self.client.service.OpenChannel())
            print('FileBet client.OpenChannel() response = %s' % self.balance)

        # if client status is 100/120 then client is ready for betting and Initialize() doesn't need to be called
        elif self.status == '100/120':
            self.update_balance()

        else:
            raise Exception('FileBet client.CheckStatus() unexpected response..')

        print('FileBet client ready for wagering..')

    def _line_to_list(self, line):
        """ internal f() to convert a line of the response to GetPrograms() to a list"""
        df_line = [x for x in line.split(' ') if len(x)]
        # if there are 5 elements, assume the "Long Name" should be combined
        if len(df_line) == 5:
            df_line = [df_line[0], df_line[1] + df_line[2], df_line[3], df_line[4]]
        elif len(df_line) == 6:
            df_line = [df_line[0], df_line[1] + df_line[2] + df_line[3], df_line[4], df_line[5]]
        elif len(df_line) == 7:
            df_line = [df_line[0], df_line[1] + df_line[2] + df_line[3] + df_line[4], df_line[5], df_line[6]]
        elif len(df_line) == 8:
            df_line = [df_line[0], df_line[1] + df_line[2] + df_line[3] + df_line[4] + df_line[5], df_line[6], df_line[7]]
        return df_line

    def close(self):
        """ close the web service after placing bets """
        resp = self.client.service.CloseChannel()
        print('FileBet client.CloseChannel response = %s' % resp)

    def update_balance(self):
        """get current wagering balance of master account on tote"""
        # TODO FileBet docs say this method should be called every 15 minutes throughout day (huey task)
        # TODO Remove hard-coded account and pin
        # 800421505 1598
        if url == WSDL_DEV:
            resp = self.client.service.UpdateBalance('123456789', '1234')  # account number and pin
            self.balance = float(resp)
        elif url == WSDL_PROD:
            resp = self.client.service.UpdateBalance('800679740', '1584')  # account number and pin
            self.balance = float(resp)

        print('FileBet client.UpdateBalance response = %s' % resp)

    def check_status(self):
        """call this method to check the health of the FileBet service"""
        # TODO FileBet docs best practices say this method should be called every 30 seconds throughout day (huey task)
        resp = self.client.service.CheckStatus()  # account number and pin
        # The return value is in the form of ‘n/m’ where n is the number of sub-accounts opened and m is the maximum number of channels.
        self.status = resp
        print('FileBet client.CheckStatus response = %s' % resp)

    def get_programs(self):
        """ get the programs available for betting from the web service
            Returns:
                DataFrame of the response
        """
        # call the web service
        response = self.client.service.GetPrograms()

        # convert the response to GetPrograms() to a DataFrame
        df = DataFrame()
        header = ['Name', 'LongName', 'Race', 'MTP']
        for i, line in enumerate(str(response).splitlines()):
            # skip the first row of headers and assume our headers are correct
            # TODO: raise an error if the headers are wrong / incorrect # of columns
            if i == 0:
                continue
            line_list = self._line_to_list(line)
            df = df.append(DataFrame([line_list], columns=header), ignore_index=True)
        return df

    def submit_bets(self, df_bets):
        """ place bets via the BetFile web service
            Args:
                df_bets: DataFrame of bets to place with columns
            Returns:
                string of the response
        """

        # validate correct df format
        self._validate_df_bets(df_bets)

        # make bet string for sending to tote
        self._make_bet_string(df_bets)

        # submit bet string to tote
        response = self.client.service.SubmitBets(self.bet_string)

        # update account balance after betting
        self.update_balance()

        # note this returning the RAW response from tote
        return response

    def _make_bet_string(self, df_bets):
        """
        helper function for making bet file / bet_string
        The requested file of bets is a single text argument to SubmitBets formatted as a collection of records separated by the newline symbol (0x0A).
        The fields within a record are separated with the pipe "|" symbol.  The bet record input format is as follows:
            a. Unique identifier to be reflected back to operator in the bet response and recorded in the AR file/audit feed.
            b. ITW event code of the track.
            c. Race
            d. Amount
            e. Bet type as defined by the GWS documentation
            f. Runners formatted according the GWS documentation rules.
        """
        # this is to prevent re-sending the same bets - if FileBet is instantiated once and then submit_bets is called more than once, the bet_string be appended to
        self.bet_string = ''

        # create a string containing the information for N bets (for N rows of the DataFrame)
        for idx, row in df_bets.iterrows():
            # convert the DataFrame to text, separated by | delimiter
            bet_line = '%s|%s|%s|%0.1f|%s|%s\n' % (row['unique_id'], row['itsp_track_sym'], row['race_num'], row['bet_amount'], row['bet_type'], row['runners'])
            # append to the string of all bets
            self.bet_string = '%s%s' % (self.bet_string, bet_line)

    def _validate_df_bets(self, df_bets):
        """
        validates the DataFrame input is of the correct format for submitting bets
        :param df_bets: DataFrame
        """
        # make sure df_bets has correct columns for making a bet string
        # TODO soon validation should only check if df_bets is an attribute of Bets() Class
        expected_cols = ['unique_id', 'itsp_track_sym', 'race_num', 'bet_amount', 'bet_type', 'runners']
        df_cols = df_bets.columns
        missing_cols = set(expected_cols) - set(df_cols)
        if len(missing_cols) > 0:
            raise Exception('FileBet.submit_bets() ERROR input DataFrame missing columns: %s' % missing_cols)
