REG_CHECK_NUMBER = r"\d*\.?\d*"  # regular expression that checks if the expression is a number.
REG_CHECK_HIGHER_LETTER = "[A-Z]"  # regular expression that checks if the expression is a big letter.
# regular expression thar check if the expression is a time in format 18:23:23213
REG_CHECK_TIME = fr"\d{{2}}:\d{{2}}:{REG_CHECK_NUMBER}"
REG_CHECK_DATE = r"\d{2}/\d{2}/\d{2}"  # regular expression thar check if the expression is a time in format is a date
REG_IGNORE_SPACE = r"\s*"
REG_widow_FILE = {'identifier_reg': "^[$]GPGGA",  # should be always &GPGGA
                'time_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",  # hhmmss.sss in UTC
                'latitude_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",
                'N/S_reg': "[NS]",
                'longitude_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",
                'E/W_reg': "[EW]",
                'fix_quality_reg': f"{REG_IGNORE_SPACE}[1-5]",  # integer between 1-5
                'num_Satellites_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",
                # number of satellites the gps have in its line of sight
                'hdop_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",  # horizontal dilution of precision
                'altitude_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",
                'altitude_units_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_HIGHER_LETTER}",  # usually meters, M
                'geoid_separation_reg': REG_CHECK_NUMBER,
                'separation_units_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_HIGHER_LETTER}",  # usually meters, M
                'age_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",  # time in seconds since last DGPA update
                # usually here goes another field, the DGPA reference station ID
                'checksum_reg': f"{REG_IGNORE_SPACE}[0-9A-Z]*[*][0-9A-Z]*",
                # now followes two lines that are not part of the gpgga massage
                'magnetic_field_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}",  # in nanoTesla [nT]
                'signal_strenght_reg': f"{REG_IGNORE_SPACE}{REG_CHECK_NUMBER}"
                }
