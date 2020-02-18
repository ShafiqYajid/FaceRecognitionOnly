
// Initialise Pusher

const pusher = new Pusher('58eec546d8492ceb70ab', {
    cluster: 'ap1',
    encrypted: true
});

var channel = pusher.subscribe('table');

channel.bind('new-record', (data) => {

   $('#namesAll').append(`
        <tr id="${data.data.id}">
            <th scope="row"> ${data.data.nameA} </th>
            <td> ${data.data.positionA} </td>
            <td> ${data.data.status} </td>
            <td> ${data.data.dateTime} </td>
        </tr>

   `)
});

channel.bind('update-record', (data) => {

    $(`#${data.data.id}`).html(`
        <th scope="row"> ${data.data.nameA} </th>
        <td> ${data.data.positionA} </td>
        <td> ${data.data.status} </td>
        <td> ${data.data.dateTime} </td>
    `)

 });