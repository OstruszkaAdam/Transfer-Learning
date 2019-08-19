import xlsxwriter

def saveArrayAsExcel(poleKzapisu, nazevSouboru):
    # Some data we want to write to the worksheet.

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(nazevSouboru + ".xlsx")
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0
    # Iterate over the data and write it out row by row.
    for item, cost in (poleKzapisu):
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, cost)
        row += 1
    # # Write a total using a formula.
    # worksheet.write(row, 0, 'Total')
    # worksheet.write(row, 1, '=SUM(B1:B4)')
    workbook.close()


def zapsatPolePoliJakoExcel(PoleZahlavi, poleKzapisu, nazevSouboru):
    # TODO nastavit jako desetinny oddelovat carku namisto tecky

    # TODO vypis pravdepodobnosti pro jednotlive kategorie jednotne radit a naformatovat = dostat kategorie do zahlavi a do tabulky jen procenta

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(nazevSouboru + ".xlsx")
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})

    # Oddeleny zapis zahlavi tabulky
    for sloupecKzapisu in range(len(PoleZahlavi)):
        worksheet.write(0, sloupecKzapisu, PoleZahlavi[sloupecKzapisu], bold)
    # end for

    # Zapis vysledku testu
    for radekKzapisu in range(len(poleKzapisu)):
        for sloupecKzapisu in range(len(poleKzapisu[radekKzapisu])):
            worksheet.write(radekKzapisu +1, sloupecKzapisu, poleKzapisu[radekKzapisu][sloupecKzapisu])
        # end for
    # end for

    workbook.close()